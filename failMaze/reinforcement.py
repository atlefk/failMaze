import sys
import numpy as np
import math
import random

import gym
import gym_maze


class ReinforcementClass(object):

    def __init__(self, render=False):
        self.render = render

        self.MAZE_SIZE = None

        self.NUM_BUCKETS = None

        self.NUM_ACTIONS = [0, 1, 2, 3]

        # Bounds for each discrete state
        self.STATE_BOUNDS = None
        # print(STATE_BOUNDS)
        '''
        Learning related constants
        '''
        self.MIN_EXPLORE_RATE = 0.001
        self.MIN_LEARNING_RATE = 0.2
        self.DECAY_FACTOR = None

        '''
        Defining the simulation related constants
        '''
        self.NUM_EPISODES = 50000
        self.MAX_T = None
        self.STREAK_TO_END = 1
        self.SOLVED_T = None  # np.prod(MAZE_SIZE, dtype=int)
        self.DEBUG_MODE = 0
        self.RENDER_MAZE = render

        '''
        Creating a Q-Table for each state-action pair
        '''
        self.q_table = None  # np.zeros(NUM_BUCKETS + (NUM_ACTIONS.__len__(),), dtype=float)

        self.env = None

    def simulate(self):

        # Instantiating the learning related parameters
        learning_rate = self.get_learning_rate(0)
        explore_rate = self.get_explore_rate(0)
        discount_factor = 0.99

        num_streaks = 0

        if self.render:
            self.env.render()

        for episode in range(self.NUM_EPISODES):

            # Reset the environment
            obv = np.zeros(2)

            # the initial state
            state_0 = self.state_to_bucket(obv)
            total_reward = 0

            for t in range(self.MAX_T):

                # Select an action
                action = self.select_action(state_0, explore_rate)

                # execute the action
                obv, reward, done, _ = self.env.step(action)
                # asd = np.array((1, 1))
                # print(obv)
                # Observe the result
                state = self.state_to_bucket(obv)
                total_reward += reward

                # Update the Q based on the result
                best_q = np.amax(self.q_table[state])
                # print(state_0)
                # print(action)
                # print(best_q)
                # print(reward)
                self.q_table[state_0 + (action,)] += learning_rate * (
                reward + discount_factor * (best_q) - self.q_table[state_0 + (action,)])

                # Setting up for the next iteration
                state_0 = state

                # Print data
                if self.DEBUG_MODE == 2:
                    print("\nEpisode = %d" % episode)
                    print("t = %d" % t)
                    print("Action: %d" % action)
                    print("State: %s" % str(state))
                    print("Reward: %f" % reward)
                    print("Best Q: %f" % best_q)
                    print("Explore rate: %f" % explore_rate)
                    print("Learning rate: %f" % learning_rate)
                    print("Streaks: %d" % num_streaks)
                    print("")

                elif self.DEBUG_MODE == 1:
                    if done or t >= self.MAX_T - 1:
                        print("\nEpisode = %d" % episode)
                        print("t = %d" % t)
                        print("Explore rate: %f" % explore_rate)
                        print("Learning rate: %f" % learning_rate)
                        print("Streaks: %d" % num_streaks)
                        print("Total reward: %f" % total_reward)
                        print("")

                # Render tha maze
                if self.render:
                    self.env.render()

                    # if env.is_game_over():
                    # sys.exit()

                if done:
                    print("Episode %d finished after %f time steps with total reward = %f (streak %d)."
                          % (episode, t, total_reward, num_streaks))

                    if t <= self.SOLVED_T:
                        num_streaks += 1
                        self.env.reset()
                    else:
                        num_streaks = 0
                    break

                elif t >= self.MAX_T - 1:
                    print("Episode %d timed out at %d with total reward = %f."
                          % (episode, t, total_reward))

            # It's considered done when it's solved over 120 times consecutively
            if num_streaks > self.STREAK_TO_END:
                break

            # Update parameters
            explore_rate = self.get_explore_rate(episode)
            learning_rate = self.get_learning_rate(episode)

    def select_action(self, state, explore_rate):
        # Select a random action
        if random.random() < explore_rate:
            actions = [0, 1, 2, 3]
            action = random.choice(actions)
        # Select the action with the highest q
        else:
            action = int(np.argmax(self.q_table[state]))
        return action

    def get_explore_rate(self, t):
        return max(self.MIN_EXPLORE_RATE, min(0.8, 1.0 - math.log10((t + 1) / self.DECAY_FACTOR)))

    def get_learning_rate(self, t):
        return max(self.MIN_LEARNING_RATE, min(0.8, 1.0 - math.log10((t + 1) / self.DECAY_FACTOR)))

    def state_to_bucket(self, state):
        bucket_indice = []
        for i in range(len(state)):
            if state[i] <= self.STATE_BOUNDS[i][0]:
                bucket_index = 0
            elif state[i] >= self.STATE_BOUNDS[i][1]:
                bucket_index = self.NUM_BUCKETS[i] - 1
            else:
                # Mapping the state bounds to the bucket array
                bound_width = self.STATE_BOUNDS[i][1] - self.STATE_BOUNDS[i][0]
                offset = (self.NUM_BUCKETS[i] - 1) * self.STATE_BOUNDS[i][0] / bound_width
                scaling = (self.NUM_BUCKETS[i] - 1) / bound_width
                bucket_index = int(round(scaling * state[i] - offset))
            bucket_indice.append(bucket_index)
        return tuple(bucket_indice)

    def start(self, env_list):
        for maze in env_list:
            self.env = gym.make(maze)
            self.MAZE_SIZE = (self.env.observation_space[0], self.env.observation_space[1])
            # print(MAZE_SIZE)
            self.NUM_BUCKETS = self.MAZE_SIZE  # one bucket per grid

            # Bounds for each discrete state
            self.STATE_BOUNDS = [(0, self.env.observation_space[0] - 1), (0, self.env.observation_space[1] - 1)]
            # print(STATE_BOUNDS)
            '''
            Learning related constants
            '''
            # self.MIN_EXPLORE_RATE = 0.001
            # self.MIN_LEARNING_RATE = 0.2
            self.DECAY_FACTOR = np.prod(self.MAZE_SIZE, dtype=float) / 10.0

            '''
            Defining the simulation related constants
            '''
            # self.NUM_EPISODES = 50000
            self.MAX_T = np.prod(self.MAZE_SIZE, dtype=int) * 100
            # self.STREAK_TO_END = 100
            self.SOLVED_T = np.prod(self.MAZE_SIZE, dtype=int)
            DEBUG_MODE = 0
            '''
            Creating a Q-Table for each state-action pair
            '''
            self.q_table = np.zeros(self.NUM_BUCKETS + (self.NUM_ACTIONS.__len__(),), dtype=float)
            '''
            Begin simulation
            '''
            #   recording_folder = "/tmp/maze_q_learning"

            #  if ENABLE_RECORDING:
            #     env.monitor.start(recording_folder, force=True)

            self.simulate()
