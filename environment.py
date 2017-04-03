""" This code is based on https://github.com/devsisters/DQN-tensorflow """
import gym
import random
import numpy as np
from skimage import transform
from skimage import color

class Environment(object):
    def __init__(self, name='Breakout-v0', width=84, height=84, history=4):
        self._env = gym.make(name)
        self._env = self._env.unwrapped
        self._env.reset()
        self._width = width
        self._height = height
        self._history = history
        self._reward = 0
        self._terminal = True
        self._screen = None
        self._screen_ori = None

    @property
    def action_size(self):
        return self._env.action_space.n
    
    @property
    def state(self):
        return self._screen, self._reward, self._terminal

    @property
    def screen(self):
        return self._screen_ori

    @property
    def lives(self):
        return self._env.ale.lives()

    def new_random_game(self, force=False):
        if self.lives == 0:
            self._env.reset()
        if force:
            self._env.reset() # avoid gym internal error
        for _ in xrange(random.randint(0, 29)):
            self._step(0)
        return self._screen, 0, 0, self._terminal

    def act(self, action, is_train=True):
        start_lives = self.lives
        self._step(action)
        if is_train and start_lives > self.lives:
            self._reward -= 1
            self._terminal = True
        return self.state

    def _step(self, action):
        self._screen_ori, self._reward, self._terminal, _ = self._env.step(action)
        self._screen = transform.resize(self._screen_ori, [self._height, self._width])
        self._screen = color.rgb2gray(self._screen)

class ReplayMemory(object):
    def __init__(self, env, capacity, batch_size):
        self._history = env._history
        self._height = env._height
        self._width = env._width
        self._capacity = capacity
        self._batch_size = batch_size
        self._actions = np.empty(self._capacity, dtype=np.int)
        self._rewards = np.empty(self._capacity, dtype=np.float)
        self._screens = np.empty((self._capacity, self._height, self._width), dtype=np.float)
        self._terminals = np.empty(self._capacity, dtype=np.float)
        self._count = 0
        self._current = 0
        self._prestat = np.empty((self._batch_size, self._history) + (self._height, self._width), dtype = np.float)
        self._poststat = np.empty((self._batch_size, self._history) + (self._height, self._width), dtype = np.float)
        print 'replay memory'

    def add(self, screen, reward, action, terminal):
        self._actions[self._current] = action
        self._rewards[self._current] = reward
        self._screens[self._current, ...] = screen
        self._terminals[self._current] = terminal
        self._count = max(self._count, self._current + 1)
        self._current = (self._current + 1) % self._capacity

    def sample(self):
        # sample random indexes
        indexes = []
        while len(indexes) < self._batch_size:
            # find random index 
            while True:
                # sample one index (ignore states wraping over 
                index = random.randint(self._history, self._count - 1)
                # if wraps over current pointer, then get new one
                if index >= self._current and index - self._history < self._current:
                    continue
                # if wraps over episode end, then get new one
                # NB! poststate (last screen) can be terminal state!
                if self._terminals[(index - self._history):index].any():
                    continue
                # otherwise use this index
                break
            # NB! having index first is fastest in C-order matrices
            self._prestat[len(indexes), ...] = self._get_state(index - 1)
            self._poststat[len(indexes), ...] = self._get_state(index)
            indexes.append(index)
        actions = self._actions[indexes]
        rewards = self._rewards[indexes]
        terminals = self._terminals[indexes]
        return self._prestat, actions, rewards, self._poststat, terminals

    def _get_state(self, index):
        index = index % self._count
        # if is not in the beginning of matrix
        if index >= self._history - 1:
            # use faster slicing
            return self._screens[(index - (self._history - 1)):(index + 1), ...]
        else:
            # otherwise normalize indexes and use slower list based access
            indexes = [(index - i) % self._count for i in reversed(range(self._history))]
            return self._screens[indexes, ...]

class History(object):
    def __init__(self, env):
        self._history = env._history
        self._height = env._height
        self._width = env._width
        self._input = np.zeros([self._history, self._height, self._width])

    @property
    def get(self):
        return self._input

    @property
    def reset(self):
        self._input *= 0

    def add(self, screen):
        self._input[:-1] = self._input[1:]
        self._input[-1] = screen

if __name__ == '__main__':
    env = Environment()
    hist = History(env)
    mem = ReplayMemory(env, 1000000, 32)
    cul_re = 0
    for i in xrange(10000):
        ac = 0 #env._env.action_space.sample()
        sc, re, ter = env.act(ac)
        mem.add(sc, re, ac, ter)
        #sc, re, ter = env.state
        hist.add(sc)
        if env.lives == 0:
            print 'reset'
            env._env.reset()
            cul_re = 0
            #import ipdb
            #ipdb.set_trace()
        if re > 0:
            cul_re += re
            print cul_re, ter, env.lives
