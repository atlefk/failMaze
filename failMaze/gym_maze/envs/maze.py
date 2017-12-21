# -*- coding: utf-8 -*-
import random
from queue import PriorityQueue

import pygame
import numpy as np
import scipy.misc
import pickle as pl


class Maze(object):
    def __init__(self, width=15, height=15, complexity=1, density=1, np_rng=np.random.RandomState(),
                 rng=random.Random()):
        """
        Creates a new maze with the given sizes, with all walls standing.
        """
        self.np_rng = np_rng
        self.rng = rng

        self.maze = self.generate2(width, height, complexity, density)
        self.maze = self.maze.reshape((self.maze.shape[0], self.maze.shape[1], 1))
        self.width = self.maze.shape[0]
        self.height = self.maze.shape[1]

    def validOperation(self, x, y, visited, width, height):

        if x < 0 or y < 0 or x >= width or y >= height or visited.__contains__((x, y)):
            return False
        else:
            return True

    def theChoices(self, x, y, visited, width, height, dirs):
        dir = self.rng.choice(dirs)

        # Break top
        if dir == 0:
            if self.validOperation(x, y - 1, visited, width, height):
                return (100, x, y - 1, 10000, 0)
            else:
                dirs.remove(0)
                return self.theChoices(x, y, visited, width, height, dirs)
        # høyre
        elif dir == 1:
            if self.validOperation(x + 1, y, visited, width, height):
                return (1000, x + 1, y, 100000, 1)
            else:
                dirs.remove(1)
                return self.theChoices(x, y, visited, width, height, dirs)
        # bunn
        elif dir == 2:
            if self.validOperation(x, y + 1, visited, width, height):
                return (10000, x, y + 1, 100, 2)
            else:
                dirs.remove(2)
                return self.theChoices(x, y, visited, width, height, dirs)
        # venstre
        elif dir == 3:
            if self.validOperation(x - 1, y, visited, width, height):
                return (100000, x - 1, y, 1000, 3)
            else:
                dirs.remove(3)
                return self.theChoices(x, y, visited, width, height, dirs)

    def theChoices2(self, x, y, visited, width, height, dirs):
        dir = self.rng.choice(dirs)

        # Break top
        if dir == 0:
            if self.validOperation(x, y - 1, visited, width, height):
                return (3, x, y - 1, 5, 0)
            else:
                dirs.remove(0)
                return self.theChoices2(x, y, visited, width, height, dirs)
        # høyre
        elif dir == 1:
            if self.validOperation(x + 1, y, visited, width, height):
                return (4, x + 1, y, 6, 1)
            else:
                dirs.remove(1)
                return self.theChoices2(x, y, visited, width, height, dirs)
        # bunn
        elif dir == 2:
            if self.validOperation(x, y + 1, visited, width, height):
                return (5, x, y + 1, 3, 2)
            else:
                dirs.remove(2)
                return self.theChoices2(x, y, visited, width, height, dirs)
        # venstre
        elif dir == 3:
            if self.validOperation(x - 1, y, visited, width, height):
                return (6, x - 1, y, 4, 3)
            else:
                dirs.remove(3)
                return self.theChoices2(x, y, visited, width, height, dirs)

    def pathFinder(self, x, y, visited, width, height):

        if self.validOperation(x + 1, y, visited, width, height):
            return True
        elif self.validOperation(x - 1, y, visited, width, height):
            return True
        elif self.validOperation(x, y + 1, visited, width, height):
            return True
        elif self.validOperation(x, y - 1, visited, width, height):
            return True
        else:
            return False

    def breakWalls2(self, Z, width, height):
        visited = []
        finished = False

        # Selects random starting point.
        x, y = (self.rng.randint(0, width - 1), self.rng.randint(0, height - 1))
        visited.append((x, y))

        dirs = [0, 1, 2, 3]
        wall = self.theChoices2(x, y, visited, width, height, dirs)
        # instead of subtracting I need to flip the bit at position wall[0]
        # Z[y, x] -= wall[0]
        Z[y, x] = self.toggleBit(Z[y, x], wall[0] - 1)
        visited.append((wall[1], wall[2]))
        Z[wall[2], wall[1]] = self.toggleBit(Z[wall[2], wall[1]], wall[3] - 1)

        superVisited = visited.copy()
        while not finished:
            dirs = [0, 1, 2, 3]
            if superVisited.__len__() == 0:
                finished = True
                break
            # Selects the newest entry in supervisited.
            x, y = superVisited[superVisited.__len__() - 1]
            if self.pathFinder(x, y, visited, width, height):
                wall = self.theChoices2(x, y, visited, width, height, dirs)
                visited.append((wall[1], wall[2]))
                superVisited.append((wall[1], wall[2]))
                Z[y, x] = self.toggleBit(Z[y, x], wall[0] - 1)
                Z[wall[2], wall[1]] = self.toggleBit(Z[wall[2], wall[1]], wall[3] - 1)
            else:
                # Removes that entry (No possible actions allowed from that entry)
                superVisited.remove((x, y))

        return Z

    def breakWalls(self, Z, width, height):
        visited = []
        finished = False

        # Selects random starting point.
        x, y = (random.randint(0, width - 1), random.randint(0, height - 1))
        visited.append((x, y))

        dirs = [0, 1, 2, 3]
        wall = self.theChoices(x, y, visited, width, height, dirs)
        Z[y, x] -= wall[0]
        visited.append((wall[1], wall[2]))
        Z[wall[2], wall[1]] -= wall[3]

        superVisited = visited.copy()
        while not finished:
            dirs = [0, 1, 2, 3]
            if superVisited.__len__() == 0:
                finished = True
                break
            # Selects the newest entry in supervisited.
            x, y = superVisited[superVisited.__len__() - 1]
            if self.pathFinder(x, y, visited, width, height):
                wall = self.theChoices(x, y, visited, width, height, dirs)

                visited.append((wall[1], wall[2]))
                superVisited.append((wall[1], wall[2]))
                Z[y, x] -= wall[0]
                Z[wall[2], wall[1]] -= wall[3]
            else:
                # Removes that entry (No possible actions allowed from that entry)
                superVisited.remove((x, y))

        return Z

    def generate2(self, width=10, height=10, complexity=2.75, density=.5):

        shape = (height, width)

        Z = np.zeros(shape, dtype=np.int)
        # 4
        ######
        # 32 #    # #8
        ######
        # 16
        # Fill borders

        for x in np.nditer(Z, op_flags=['readwrite']):
            x[...] = 0x3C
            # 32+16+8+4+2+1
        Z = self.breakWalls2(Z, width, height)
        return Z

    def toggleBit(self, int_type, offset):
        mask = 1 << offset
        return (int_type ^ mask)

    def generate(self, width=10, height=10, complexity=2.75, density=.5):
        try:
            Z = pl.load(open(str(width) + 'x' + str(height) + '.p', 'rb'))
            return Z
        except:
            shape = (height, width)

            Z = np.zeros(shape, dtype=np.int)
            # 100
            ######
            # 100000    #    # #1000
            ######
            # 10000
            # Fill borders

            for x in np.nditer(Z, op_flags=['readwrite']):
                x[...] = 111100

            Z = self.breakWalls(Z, width, height)
            pl.dump(Z, open(str(width) + 'x' + str(height) + '.p', 'wb'))
            return Z


class ActionSpace:
    def __init__(self):
        self.shape = 4

    @staticmethod
    def sample():
        return random.randint(0, 3)


class StateSpace:
    def __init__(self, game):
        self.shape = game._get_state().shape


class MazeGame(object):
    """
    Class for interactively playing random maze games.
    """

    def __init__(self,
                 width,
                 height,
                 screen_width=640,
                 screen_height=480,
                 state_representation="image",
                 image_state_width=80,
                 image_state_height=80,
                 seed=None,
                 seed_both=False,
                 brute=False,
                 reinforcement=False
                 ):

        # Pygame
        pygame.init()

        pygame.font.init()
        pygame.display.set_caption("DeepMaze")
        self.width = width
        self.height = height
        self.font = pygame.font.SysFont("Arial", size=16)
        self.screen = pygame.display.set_mode((screen_width + 5, screen_height + 5), 0, 32)
        self.screen_width = screen_width
        self.screen_height = screen_height

        # self.surface = pygame.Surface(self.screen.get_size())

        # self.surface = self.surface.convert()
        # self.surface.fill((255, 255, 255))
        self.txt_up = self.font.render("U", False, (0, 0, 0))
        self.txt_down = self.font.render("D", False, (0, 0, 0))
        self.txt_left = self.font.render("L", False, (0, 0, 0))
        self.txt_right = self.font.render("R", False, (0, 0, 0))
        self.tile_w = (screen_width + 4) / width
        self.tile_h = (screen_height + 4) / height

        # Game Parameters
        self.width = width

        self.height = height
        self.terminal = False

        # State Definition
        self.state_representation = state_representation
        self.image_state_size = (image_state_width, image_state_height)

        # Random Seeding
        self.seed = seed

        self.seed_both = seed_both

        self.rng = random.Random(self.seed if self.seed_both else None)
        self.np_rng = np.random.RandomState(self.seed)

        # Maze Generation
        self.maze = None
        self.optimal_path = None
        self.optimal_path_length = None

        # Players
        self.player, self.target = None, None

        self.background = None
        self.maze_layer = None

        self.mazeImage = None

        # Reset
        self.reset()
        # q-stuff
        self.q_table = np.zeros(shape=(self.maze.maze.shape[0], self.maze.maze.shape[1]), dtype=np.int8)
        self.q_table.fill(-1)

        self.brute = brute

        self.reinforcement = reinforcement

    @staticmethod
    def rgb2gray(rgb):
        return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])

    def get_state(self):

        if self.state_representation == "array":
            state = np.array(self.maze.maze, copy=True)

            state[self.player[1], self.player[0], 0] = self.toggleBit(state[self.player[1], self.player[0], 0], 0)
            state[self.target[1], self.target[0], 0] = self.toggleBit(state[self.target[1], self.target[0], 0], 1)

            return state

        elif self.state_representation == "normal":

            state = np.array(self.maze.maze, copy=True)
            self.createArrayOfBoard()

            return state

    def reset(self):
        # Reinitialize RNG
        self.rng = random.Random(self.seed if self.seed_both else None)
        self.np_rng = np.random.RandomState(self.seed)

        # Reset terminal state
        self.terminal = False

        # Create new maze
        self.maze = Maze(self.width, self.height, np_rng=self.np_rng, rng=self.rng)

        # Set player positions
        self.player, self.target = self.spawn_players()

        # Create a background
        self.background = pygame.Surface(self.screen.get_size()).convert()
        self.background.fill((255, 255, 255))

        # Create a layer for the maze
        self.maze_layer = pygame.Surface(self.screen.get_size()).convert_alpha()
        self.maze_layer.fill((0, 0, 0, 0,))

        self._drawMaze()

        return self.get_state()

    def __draw_player(self, colour=(0, 0, 150), transparency=255):

        x = int(self.player[0] * self.tile_w + self.tile_w * 0.5 + 0.5)
        y = int(self.player[1] * self.tile_h + self.tile_h * 0.5 + 0.5)
        r = int(min(self.tile_w, self.tile_h) / 5 + 0.5)

        pygame.draw.circle(self.maze_layer, colour + (transparency,), (x, y), r)

    def __colour_cell(self, colour, transparency, cell=-1):
        if cell == -1:
            cell = self.target
        if not (isinstance(cell, (list, tuple, np.ndarray)) and len(cell) == 2):
            raise TypeError("cell must a be a tuple, list, or numpy array of size 2")

        x = int(cell[0] * self.tile_w + self.tile_w * 0.10)
        y = int(cell[1] * self.tile_h + self.tile_h * 0.10)
        w = int(self.tile_w * 0.8 + 0.5 - 1)
        h = int(self.tile_h * 0.8 + 0.5 - 1)
        pygame.draw.rect(self.maze_layer, colour + (transparency,), (x, y, w, h))

    def testBit(self, int_type, offset):
        mask = 1 << offset
        return (int_type & mask)

    def toggleBit(self, int_type, offset):
        mask = 1 << offset
        return (int_type ^ mask)

    def _drawMaze(self):
        # drawing the horizontal lines
        line_colour = (0, 0, 0, 255)
        for y in range(len(self.maze.maze)):
            for x in range(len(self.maze.maze[y])):
                value = self.maze.maze[y][x]
                # Checks if there is a line on the leftside.
                # 6-1
                if self.testBit(value, 5) == 32:
                    pygame.draw.line(self.maze_layer, line_colour, (x * self.tile_w, y * self.tile_h),
                                     (x * self.tile_w, y * (self.tile_h) + self.tile_h))
                # Checks if there is a line on the bottom.
                # 5-1
                if self.testBit(value, 4) == 16:
                    pygame.draw.line(self.maze_layer, line_colour, (x * self.tile_w, y * (self.tile_h) + self.tile_h),
                                     (x * (self.tile_w) + self.tile_w, y * (self.tile_h) + self.tile_h))
                # Checks if there is a line on the right side.
                if self.testBit(value, 3) == 8:
                    pygame.draw.line(self.maze_layer, line_colour, (x * self.tile_w + self.tile_w, y * self.tile_h),
                                     (x * (self.tile_w) + self.tile_w, y * (self.tile_h) + self.tile_h))

                # Checks if there is a line on the top.
                if self.testBit(value, 2):
                    pygame.draw.line(self.maze_layer, line_colour, (x * self.tile_w, y * self.tile_h),
                                     (x * (self.tile_w) + self.tile_w, y * self.tile_h))

        self.__colour_cell(colour=(150, 0, 0), transparency=235)
        self.__draw_player()
        self.screen.blit(self.maze_layer, (0, 0))

        pygame.display.flip()

    def dfs(self, start, goal):
        stack = [(start, [start])]

        possible_path = PriorityQueue()

        while stack:
            (vertex, path) = stack.pop()
            legal_cells = set(self.legal_directions(*vertex)) - set(path)
            for next in legal_cells:
                if next == goal:
                    full_path = path + [next]
                    length = len(path)
                    possible_path.put((length, full_path))
                else:
                    stack.append((next, path + [next]))

        return possible_path.get()

    def spawn_players(self):
        randomPos = np.where(self.maze.maze >= 0)

        possiblePlaces = np.array([(randomPos[0][i], randomPos[1][i]) for i in range(len(randomPos[0]))])

        target_spawn = tuple(self.rng.choice(possiblePlaces))
        target_spawn = (target_spawn[0], target_spawn[1])
        #print(target_spawn)
        while True:  # todo xd
            player_spawn = tuple(self.rng.choice(possiblePlaces))
            #player_spawn = (player_spawn[0], player_spawn[1])
            #player_spawn = (target_spawn[0]-1, player_spawn[1]+2)
            #old = player_spawn
            #player_spawn = target_spawn
            #target_spawn = (old[0], old[1]-1)
            if target_spawn != player_spawn:
                break
        # Todo
        self.optimal_path_length, self.optimal_path = self.dfs(player_spawn, target_spawn)

        return player_spawn, target_spawn

    def createArrayOfBoard(self):

        self.mazeImage = self.background.copy()
        self.mazeImage.blit(self.maze_layer, (0, 0))
        arr = pygame.surfarray.array3d(self.mazeImage)
        self.image_state_size = (128, 128)
        #arr = np.fliplr(arr)
        """if random.random() < 1:
            arr = scipy.misc.imresize(arr, self.image_state_size)
            scipy.misc.imsave('outfile'+str(self.player)+'.jpg', arr)
        else:
            arr = scipy.misc.imresize(arr, self.image_state_size)
        """

        arr = scipy.misc.imresize(arr, self.image_state_size)
        arr = arr / 255

        self.mazeImage = arr

    def render(self):
        try:
            for (x, y, z), value in np.ndenumerate(self.maze.maze):
                pos = (x * self.tile_w, y * self.tile_h, self.tile_w + 1, self.tile_h + 1)
                # print(pos)
                txt_type = self.q_table[y, x]
                # print(self.q_table)
                # print(txt_type)
                if txt_type == 1:
                    self.maze_layer.blit(self.txt_up, (pos[0] + 8, pos[1] + 8))  # Up
                if txt_type == 2:
                    self.maze_layer.blit(self.txt_right, (pos[0] + 8, pos[1] + 8))  # Right
                if txt_type == 3:
                    self.maze_layer.blit(self.txt_down, (pos[0] + 8, pos[1] + 8))  # Down
                if txt_type == 4:
                    self.maze_layer.blit(self.txt_left, (pos[0] + 8, pos[1] + 8))  # Left
            self.screen.blit(self.background, (0, 0))
            self.screen.blit(self.maze_layer, (0, 0))
            pygame.display.flip()
        except:
            pass

    def on_return(self, reward):
        return self.get_state(), reward, self.terminal, {
            "optimal_path": self.optimal_path_length
        }

    def step(self, a):
        if self.terminal:
            return self.on_return(1)

        a_vec = self.to_action(a)
        posx, posy = self.player
        nextx, nexty = posx + a_vec[0], posy + a_vec[1]

        legal = self.is_legal(nextx, nexty)

        if legal:
            self.__draw_player(transparency=00)
            self.player = (nextx, nexty)
            self.__draw_player()

        if self.brute:
            finish = False
            if self.player == self.target:
                finish = True
            return legal, finish, self.optimal_path_length

        if self.reinforcement:
            finish = False
            if self.player == self.target:
                reward = 1
                finish = True
            else:
                reward = -0.1 / (self.maze.maze.shape[0] * self.maze.maze.shape[1])

            player = np.array((self.player[0], self.player[1]))
            return player, reward, finish, self.optimal_path_length

        if self.player == self.target:
            self.__colour_cell(colour=(150, 0, 0), transparency=235)
            self.terminal = True
            return self.on_return(1)

        return self.on_return(-0.1)

    @staticmethod
    def quit():
        try:
            pass
            # pygame.display.quit()
            # pygame.quit()
        except:
            pass

    def to_action(self, a):
        # GO UP
        if a == 0:
            return 0, -1
        # GO RIGHT
        elif a == 1:
            return 1, 0
        # GO DOWN
        elif a == 2:
            return 0, 1
        # GO LEFT
        elif a == 3:
            return -1, 0

    def legal_directions(self, posx, posy):
        legal = []

        possible_moves = [
            (posx + 0, posy + 1),  # Down
            (posx + 0, posy - 1),  # Up
            (posx + 1, posy + 0),  # Right
            (posx - 1, posy + 0)  # Left
        ]
        legal = possible_moves.copy()
        value = self.maze.maze[posy, posx, 0]
        # 100000 left
        # 10000 bottom
        # 1000 right
        # 100 top

        if self.testBit(value, 5) == 32:
            legal.remove((posx - 1, posy))
        if self.testBit(value, 4) == 16:
            legal.remove((posx, posy + 1))
        if self.testBit(value, 3) == 8:
            legal.remove((posx + 1, posy + 0))
        if self.testBit(value, 2) == 4:
            legal.remove((posx, posy - 1))

        return legal

    def is_legal(self, nextx, nexty):
        x, y = self.player
        value = self.maze.maze[y, x, 0]
        if y < nexty:
            if self.testBit(value, 4) == 16:
                return False
            else:
                #print("I go down")
                return True

        # Tries to go up
        elif y > nexty:
            if self.testBit(value, 2) == 4:
                return False
            else:
                #print("I go up")
                return True

        # Tries to go the the right
        elif x < nextx:
            if self.testBit(value, 3) == 8:
                return False
            else:
                #print("I go right")
                return True
        # Tries go to the left
        elif x > nextx:
            if self.testBit(value, 5) == 32:
                return False
            else:
                #print("I got left")
                return True


