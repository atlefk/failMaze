import gym
import gym_maze
import numpy as np
import json
from example.logger import logger
import time

class bfClass(object):
    def __init__(self, render=False):
        self.actualSteps = 0

        self.render = render


    def bruting(self, env, dir=-1, step=0):

        if self.render:
            env.render()
        action = [0, 1, 2, 3]
        step += 1
        #print(step)
        for e in action:
            self.actualSteps += 1
            if e == 0 and dir != 2:
                val = env.step(0)
                if val[1] and step == val[2]:
                    print(step)
                    return True
                if val[0]:
                    if self.bruting(env, 0, step):
                        return True
                    else:
                        env.step(2)
            if e == 1 and dir != 3:
                val = env.step(1)
                if val[1] and step == val[2]:
                    return True
                if val[0]:
                    if self.bruting(env, 1, step):
                        return True
                    else:
                        env.step(3)
            if e == 2 and dir != 0:
                val = env.step(2)
                if val[1] and step == val[2]:
                    return True
                if val[0]:
                    if self.bruting(env, 2, step):
                        return True
                    else:
                        env.step(0)

            if e == 3 and dir != 1:
                val = env.step(3)
                if val[1] and step == val[2]:
                    return True
                if val[0]:
                    if self.bruting(env, 3, step):
                        return True
                    else:
                        env.step(1)

    def brutingspec(self, env, dir=-1, step=0):
        if self.render:
            env.render()
        action = [0, 1, 2, 3]
        step += 1
        for e in action:
            self.actualSteps += 1
            if e == 0 and dir != 2:
                val = env.step(0)
                if val[1] and step == val[2]:
                    return True
                elif step > val[2]:
                    return
                if val[0]:
                    if self.brutingspec(env, 0, step):
                        return True
                    else:
                        env.step(2)
            if e == 1 and dir != 3:
                val = env.step(1)
                if val[1] and step == val[2]:
                    return True
                elif step > val[2]:
                    return
                if val[0]:
                    if self.brutingspec(env, 1, step):
                        return True
                    else:
                        env.step(3)
            if e == 2 and dir != 0:
                val = env.step(2)
                if val[1] and step == val[2]:
                    return True
                elif step > val[2]:
                    return
                if val[0]:
                    if self.brutingspec(env, 2, step):
                        return True
                    else:
                        env.step(0)

            if e == 3 and dir != 1:
                val = env.step(3)
                if val[1] and step == val[2]:
                    return True
                elif step > val[2]:
                    return
                if val[0]:
                    if self.brutingspec(env, 3):
                        return True
                    else:
                        env.step(1)

    def start(self, env_list, nTimes):
        print("LOL")
        results = dict()
        for env_name in env_list:
            env = gym.make(env_name)
            results[env_name] = []
            for j in range(nTimes):
                self.actualSteps = 0
                start_time = time.time()
                print("Creating env %s" % env_name)
                env.reset()
                self.bruting(env)
                #self.brutingspec(env)
                print(self.actualSteps)
                result = time.time()-start_time
                results[env_name].append(result)
                #print("--- %s seconds ---" % (time.time() - start_time))

        return results