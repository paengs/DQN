import random
from tqdm import tqdm
from deep_q_network import DQN
from environment import Environment
from environment import History
from environment import ReplayMemory
import torch
from torch.autograd import Variable

dtype = torch.cuda.FloatTensor

class Agent(object):
    def __init__(self, conf):
        self.env = Environment(name=conf.env, width=conf.width, height=conf.height, history=conf.history)
        self.hist = History(self.env)
        self.mem = ReplayMemory(self.env, capacity=conf.mem_capacity, batch_size=conf.batch_size)
        self._capa = conf.mem_capacity
        self._ep_en = conf.ep_end
        self._ep_st = conf.ep_start
        self._learn_st = conf.learn_start
        self._tr_freq = conf.train_freq
        self._update_freq = conf.update_freq
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
        #for self.step in xrange(50000000):
        for self.step in tqdm(range(0, 50000000), ncols=70, initial=0):
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
                    avg_reward = total_reward / 10000.
                    avg_loss = self.total_loss / self.update_count
                    avg_q = self.total_q / self.update_count
                    print '# games: {}, reward: {}, loss: {}, q: {}'.format(num_game, avg_reward, avg_loss, avg_q)
                    num_game = 0
                    total_reward = 0.
                    self.total_loss = 0.
                    self.total_q = 0.
                    self.update_count = 0
                    ep_reward = 0.
                    ep_rewards = []
                    actions = []
     
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
                if self.step % (self._update_freq * 10) == (self._update_freq*10) -1:
                    torch.save(self.target_q, 'models/model_{}'.format(self.step))
                #print 'update'
       
    def play(self, model_path, num_ep=100):
        self.q = torch.load(model_path)
        for ep in range(num_ep):
            screen, reward, action, terminal = self.env.new_random_game()
            current_reward = 0
            for _ in range(self.env._history):
                self.hist.add(screen)
            for _ in range(10000):
                action = self._select_action(test_mode=True)
                screen, reward, terminal = self.env.act(action)
                self.hist.add(screen)
                current_reward += reward
                if terminal:
                    print current_reward
                    break

    def _q_learning(self):
        sc_t, actions, rewards, sc_t_1, terminals = self.mem.sample()
        batch_obs_t = self._to_tensor(sc_t)
        batch_obs_t_1 = self._to_tensor(sc_t_1)
        batch_rewards = self._to_tensor(rewards).unsqueeze(1)
        batch_actions = self._to_tensor(actions, data_type=torch.cuda.LongTensor).unsqueeze(1)
        batch_terminals = self._to_tensor(1.-terminals).unsqueeze(1)

        q_values = self.q(batch_obs_t).gather(1, batch_actions)
        batch_obs_t_1.volatile=True
        next_max_q_values = self.target_q(batch_obs_t_1).max(1)[0]
        next_q_values = batch_terminals * next_max_q_values
        target_q_values = batch_rewards + (0.99*next_q_values)
        target_q_values.volatile=False

        cri = torch.nn.SmoothL1Loss()
        self.loss = cri(q_values, target_q_values)
        self.optim.zero_grad()
        self.loss.backward()
        self.optim.step()
        self.update_count += 1
        self.total_q += q_values.data.mean()
        self.total_loss += self.loss.data.mean()

    def _select_action(self, test_mode=False):
        # epsilon greedy policy
        if not test_mode:
            ep = self._ep_en + max(0., (self._ep_st - self._ep_en) * (self._capa - max(0., self.step - self._learn_st)) / self._capa)
        else:
            ep = -1.
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
    #agent.play()
    agent.train()
