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
    #parser = argparse.ArgumentParser(add_help=True)
    #parser.add_argument('-t', help='Select prediction mode (mts/mole)', default='mts', dest='trin')
    #parser.add_argument('-m', help='Gpu ID (if None, cpu mode)', default=0, dest='gpu_id')
    #parser.add_argument('-p', help='Whole slide image path', default=None, dest='sld_path')
    #output = parser.parse_args()
    conf = Config()
    agent = Agent(conf)

    print 'start'
