import random
from deep_q_network import DQN
from environment import Environment
from environment import History
from environment import ReplayMemory
import torch
from torch.autograd import Variable

dtype = torch.cuda.FloatTensor

class Agent(object):
    def __init__(self):
        self.env = Environment()
        self.hist = History(self.env)
        self.mem = ReplayMemory(self.env, capacity=1000000, batch_size=32)
        self._capa = 1000000
        self._ep_en = 0.1
        self._ep_st = 1.
        self._learn_st = 50000
        #self._learn_st = 4
        self._tr_freq = 4
        self._update_freq = 10000
        self.q = DQN(self.hist._history, self.env.action_size).type(dtype)
        self.target_q = DQN(self.hist._history, self.env.action_size).type(dtype)
        self.optim = torch.optim.RMSprop(self.q.parameters(), lr=0.00025, alpha=0.95, eps=0.01)

    def train(self):
        screen, reward, action, terminal = self.env.new_random_game()
        for _ in range(self.env._history):
            self.hist.add(screen)
        num_game, self.update_count, ep_reward = 0, 0, 0.
        total_reward, self.total_loss, self.total_q = 0., 0., 0.
        ep_rewards, actions = [], []
        for self.step in xrange(50000000):
            if self.step == self._learn_st:
                num_game, self.update_count, ep_reward = 0, 0, 0.
                total_reward, self.total_loss, self.total_q = 0., 0., 0.
                ep_rewards, actions = [], []

            action = self._select_action()
            screen, reward, terminal = self.env.act(action)
            self.observe(screen, reward, action, terminal)
            if terminal:
                screen, reward, action, terminal = self.env.new_random_game()
                num_game += 1
                ep_rewards.append(ep_reward)
                ep_reward = 0.
            else:
                ep_reward += reward
            actions.append(action)
            total_reward += reward
            if self.step >= self._learn_st:
                if self.step % 10000 == 10000 - 1:
                    #avg_reward = total_reward / self.test_step
                    avg_loss = self.total_loss / self.update_count
                    avg_q = self.total_q / self.update_count
                    print 'loss: {}, q: {}'.format(avg_loss, avg_q)
     
    def observe(self, screen, reward, action, terminal):
        reward = max(-1., min(1., reward))
        self.hist.add(screen)
        self.mem.add(screen, reward, action, terminal)
        if self.step > self._learn_st:
            if self.step % self._tr_freq == 0:
                self._q_learning()
                #print '{} q-learning'.format(self.step)
            if self.step % self._update_freq == self._update_freq - 1:
                self.target_q.load_state_dict(self.q.state_dict())
                #print 'update'
       
    def play(self):
        print 'play'

    def _q_learning(self):
        sc_t, actions, rewards, sc_t_1, terminals = self.mem.sample()
        batch_obs_t = self._to_tensor(sc_t)
        batch_obs_t_1 = self._to_tensor(sc_t_1)
        batch_rewards = self._to_tensor(rewards)#, data_type=torch.cuda.LongTensor)
        batch_actions = self._to_tensor(actions, data_type=torch.cuda.LongTensor)
        batch_terminals = self._to_tensor(1.-terminals)

        q_values = self.q(batch_obs_t).gather(1, batch_actions.unsqueeze(1))
        next_max_q_values = self.target_q(batch_obs_t_1).max(1)[0]
        next_q_values = batch_terminals * next_max_q_values
        target_q_values = batch_rewards + (0.99*next_q_values)
        bellman_err = target_q_values - q_values
        self.loss = bellman_err.clamp(-1, 1)
        #self.loss = clipped_bellman_err * -1.0
        
        self.optim.zero_grad()
        q_values.backward(self.loss.data.unsqueeze(1))
        #self.loss.backward()
        self.optim.step()
        self.update_count += 1
        self.total_q += q_values.data.mean()
        self.total_loss += self.loss.data.mean()

    def _select_action(self):
        # epsilon greedy policy
        ep = self._ep_en + max(0., (self._ep_st - self._ep_en) * (self._capa - max(0., self.step - self._learn_st)) / self._capa)
        if random.random() < ep:
            action = random.randrange(self.env.action_size)
        else:
            inputs = self._to_tensor(self.hist.get)
            pred = self.q(inputs.unsqueeze(0))
            action = pred.data.max(1)[1][0][0]
        return action

    def _to_tensor(self, ndarray, data_type=dtype):
        return Variable(torch.from_numpy(ndarray)).type(data_type)

if __name__ == '__main__':
    agent = Agent()
    agent.train()
    '''
    from skimage import io
    for i in range(10):
        agent.train()
        st_img = agent.hist.get[0]
        prev_img = st_img
        io.imsave('/data/test_{}.jpg'.format(i), st_img)
    import ipdb
    ipdb.set_trace()
    '''
