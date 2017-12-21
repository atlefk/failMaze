#from bruteForce import bfClass
#from reinforcement import ReinforcementClass
from mazeClass import MazeClass

if __name__ == "__main__":
    #brute = bfClass(render=True)
    #reinforcement = ReinforcementClass(render=True)
    project = MazeClass()

    env_listBrute = [
        "maze-arr-4x4-full-deterministic-b0-v0",
        "maze-arr-6x6-full-deterministic-b0-v0",
        "maze-arr-9x9-full-deterministic-b0-v0",
        "maze-arr-19x19-full-deterministic-b0-v0",
        "maze-arr-25x25-full-deterministic-b0-v0",
        "maze-arr-35x35-full-deterministic-b0-v0",
        "maze-arr-55x55-full-deterministic-b0-v0",
        #"maze-arr-90x90-full-deterministic-b0-v0"
    ]
    env_listReinforcement = [
        "maze-arr-4x4-full-deterministic-rf-v0",
        "maze-arr-6x6-full-deterministic-rf-v0",
        "maze-arr-9x9-full-deterministic-rf-v0",
        "maze-arr-19x19-full-deterministic-rf-v0",
        "maze-arr-25x25-full-deterministic-rf-v0",
        "maze-arr-35x35-full-deterministic-rf-v0",
        #"maze-arr-55x55-full-deterministic-rf-v0",
        #"maze-arr-90x90-full-deterministic-rf-v0"
    ]

    env_list = [
        "maze-normal-4x4-full-deterministic-v0"
        # maze-arr-4x4-deterministic-v0",
        #"maze-normal-6x6-full-deterministic-v0",
        # "maze-arr-7x7-full-deterministic-v0",
        # "maze-arr-11x11-stochastic-v0",
        # "maze-arr-9x9-full-deterministic-v0",
        # "maze-arr-11x11-full-deterministic-v0",
        # "maze-arr-12x12-full-deterministic-v0",
        # "maze-arr-13x13-full-deterministic-v0",
        # "maze-arr-15x15-full-deterministic-v0",
        # "maze-arr-17x17-full-deterministic-v0",
        # "maze-arr-19x19-full-deterministic-v0",
        # "maze-arr-25x25-full-deterministic-v0",
        # "maze-arr-35x35-full-deterministic-v0",
        # "maze-arr-55x55-full-deterministic-v0"
    ]

    #results = brute.start(env_list=env_listBrute, nTimes=10)
    #print(results)
    #results = reinforcement.start(env_list=env_listReinforcement, nTimes=10)
    #print(results)
    project.start(env_list)
