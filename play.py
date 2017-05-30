import argparse
from agent import Agent

class Config(object):
    env = 'Breakout-v0'
    width = 84
    height = 84
    history = 4
    mem_capacity = 1000000
    batch_size = 32
    train_freq = 4
    update_freq = 10000
    learn_start = 50000
    ep_start = 1.
    ep_end = 0.1

if __name__ == '__main__':
    conf = Config()
    agent = Agent(conf)
    agent.play('models/model_48799999')
