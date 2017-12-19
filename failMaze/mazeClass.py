import gym
import gym_maze
import numpy as np
import json
from example.logger import logger
from example.dqn.dqn_example_5 import DQN
from sklearn.preprocessing import minmax_scale

class MazeClass(object):
    def __init__(self, render=False):

        self.env = None
        self.render = render

        self.batch_size = 64
        self.epochs = 500
        self.train_epochs = 1
        self.memory_size = 10000
        self.timeout = 1500
        self.epsilon_increase = False

        self.env = None

        self.agent = None

    '''
    def preprocess(self, state, agent):
        
        new_state = np.zeros(shape=(1,) + agent.state_size)
        new_state[:1, :state.shape[0], :state.shape[1], :state.shape[2]] = state
        return new_state
    '''
    def preprocess(self, state, agent):
        new_state = np.reshape(state, state.shape[:2])
        new_state = minmax_scale(new_state)
        #print(new_state)
        new_state = np.reshape(new_state, (1, ) + new_state.shape + (1, ))
        return new_state

    def start(self, env_list):
        perfectRows = False
        while not perfectRows:
            for env_name in env_list:
                print("Creating env %s" % env_name)
                self.env = gym.make(env_name)
                self.agent = DQN(
                    self.env.observation_space,
                    self.env.action_space,
                    memory_size=self.memory_size,
                    batch_size=self.batch_size,
                    train_epochs=self.train_epochs,
                    e_min=0,
                    e_max=1.0,
                    e_steps=100000,
                    lr=0.3,
                    discount=0.99
                )
                self.agent.model.summary()
                try:
                    self.agent.load("./model_weights.h5")

                except:
                    print("cant find weights")
                victories_before_train = 50
                victories = 0
                perfect_in_row = 0
                perfects_before_next = 10
                self.agent.epsilon = self.agent.epsilon_max
                phase = "exploit"
                self.agent.save("./model_weights.h5")

                epoch = 0
                while epoch < self.epochs:
                    epoch += 1

                    # Reset environment
                    state = self.env.reset()

                    state = self.preprocess(state, self.agent)
                    #print(state)
                    terminal = False
                    timestep = 0

                    while not terminal:
                        timestep += 1

                        # Draw environment on screen
                        if self.render:
                            self.env.render()

                        # Draw action from distribution
                        action = self.agent.act(state, force_exploit=True if phase == "exploit" else False)

                        # Perform action in environment
                        next_state, reward, terminal, info = self.env.step(action)
                        next_state = self.preprocess(next_state, self.agent)

                        # Experience replay
                        self.agent.remember(state, action, reward, next_state, terminal)

                        state = next_state

                        if terminal:
                            # Terminal means victory
                            victories += 1

                            # If it a prefect round, set to test phase
                            if timestep == info["optimal_path"]:
                                phase = "exploit"
                                perfect_in_row += 1
                            else:
                                phase = "explore"
                                perfect_in_row = 0

                            break
                        elif timestep >= self.timeout:
                            if self.epsilon_increase:
                                self.agent.epsilon = min(self.agent.epsilon_max,
                                                         self.agent.epsilon + (self.agent.epsilon_decay * timestep))
                            phase = "explore"
                            perfect_in_row = 0
                            break

                    if len(self.agent.memory) > self.agent.batch_size:
                        self.agent.replay(q_table=self.env.env.q_table)

                    if self.render:
                        self.env.render()

                    logger.info(json.dumps({
                        "epoch": epoch,
                        "steps": timestep,
                        "optimal": info["optimal_path"],
                        "epsilon": self.agent.epsilon,
                        "loss": self.agent.average_loss(),
                        "terminal": terminal,
                        "replay": len(self.agent.memory),
                        "perfect_in_row": perfect_in_row,
                        "phase": phase,
                        "env": env_name
                    }))

                    if perfect_in_row >= perfects_before_next:
                        epoch = self.epochs
                        perfectRows = True
