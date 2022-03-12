from collections import namedtuple

import random
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from params import *

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

BATCH_SIZE = 16
CAPACITY = 5000


Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))


class Agent1:
    def __init__(self):
        self.brain = Brain1() 

    def update_q_function(self):
        self.brain.replay()

    def get_action(self, state, episode):
        action = self.brain.decide_action(state, episode)
        return action

    def memorize(self, state, action, state_next, reward):
        self.brain.memory.push(state, action, state_next, reward)

    def update_target_q_function(self):
        self.brain.update_target_q_network()
        
class ReplayMemory:
    def __init__(self, CAPACITY):
        self.capacity = CAPACITY 
        self.memory = []  
        self.index = 0  
    def push(self, state, action, state_next, reward):
        if len(self.memory) < self.capacity:
            self.memory.append(None) 
        self.memory[self.index] = Transition(state, action, state_next, reward)
        self.index = (self.index + 1) % self.capacity 
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    def __len__(self):
        return len(self.memory)

class Net1(nn.Module):
    def __init__(self, n_in, n_mid, n_out):
        super(Net1, self).__init__()
        self.fc1 = nn.Linear(n_in, n_mid)
        self.fc2 = nn.Linear(n_mid, n_mid)
        self.fc3 = nn.Linear(n_mid, n_out)

    def forward(self, x): # 输入state的三个信息
        h1 = F.leaky_relu(self.fc1(x))
        h2 = F.leaky_relu(self.fc2(h1))
        output = self.fc3(h2)
        return output


class Brain1:
    def __init__(self):
        self.num_actions = N_SERVER 
        self.memory = ReplayMemory(CAPACITY)
        n_in, n_mid, n_out =  (2*N_SERVER+1)*N_JOB, 32, N_SERVER
        if LOAD_OK:
            self.main_q_network = torch.load(PATH)
            self.target_q_network = torch.load(PATH) 
        else:
            self.main_q_network = Net1(n_in, n_mid, n_out).to(device) 
            self.target_q_network = Net1(n_in, n_mid, n_out).to(device) 

        self.optimizer = optim.Adam(
            self.main_q_network.parameters(), lr=0.0001)
        # print(self.main_q_network)

    def replay(self):
        if len(self.memory) < BATCH_SIZE:
            return
        self.batch, self.state_batch, self.action_batch, self.reward_batch, self.non_final_next_states = self.make_minibatch()
        self.expected_state_action_values = self.get_expected_state_action_values()
        self.update_main_q_network()

    def decide_action(self, state, episode):
        epsilon = 0.5 * (1 / (episode + 1))
        if epsilon <= np.random.uniform(0, 1):
        # if np.random.randint(1,10) >=2:
            self.main_q_network.eval() 
            with torch.no_grad():
                action = self.main_q_network(state).max(1)[1].view(1, 1).to(device)
        else:
            action = torch.LongTensor(
                [[random.randrange(self.num_actions)]]).to(device)
        return action.to(device)

    def make_minibatch(self):
        transitions = self.memory.sample(BATCH_SIZE)
        batch = Transition(*zip(*transitions))
    
        state_batch = torch.cat(batch.state)

        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        non_final_next_states = torch.cat([s for s in batch.next_state
                                           if s is not None])

        non_final_next_states = non_final_next_states
        return batch, state_batch, action_batch, reward_batch, non_final_next_states

    def get_expected_state_action_values(self):
        self.main_q_network.eval()
        self.target_q_network.eval()

        # self.state_action_values = self.main_q_network(self.state_batch).cuda(device).gather(1, self.action_batch)
        self.state_action_values = self.main_q_network(self.state_batch).gather(1, self.action_batch.to(device)).to(device)

        non_final_mask = torch.ByteTensor(tuple(map(lambda s: s is not None, self.batch.next_state)))
        next_state_values = torch.zeros(BATCH_SIZE)

        a_m = torch.zeros(BATCH_SIZE).type(torch.LongTensor).to(device)

        a_m[non_final_mask] = self.main_q_network(self.non_final_next_states).detach().max(1)[1].to(device)
        a_m_non_final_next_states = a_m[non_final_mask].view(-1, 1).to(device)

        next_state_values[non_final_mask] = self.target_q_network(self.non_final_next_states).gather(1, a_m_non_final_next_states).detach().squeeze()
        # print(self.reward_batch)
        # print(next_state_values)
        expected_state_action_values = self.reward_batch + GAMMA * next_state_values.to(device)
        return expected_state_action_values

    def update_main_q_network(self):
        self.main_q_network.train()
        loss = F.smooth_l1_loss(self.state_action_values,
                                self.expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()  
        loss.backward()  
        self.optimizer.step() 

    def update_target_q_network(self): 
        self.target_q_network.load_state_dict(self.main_q_network.state_dict())
