from collections import namedtuple

import random
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from params import *
from copy import copy  

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

BATCH_SIZE = 8
CAPACITY = 10000


Transition = namedtuple('Transition', ('state1', 'state2', 'state3', 'action', 'next_state1', 'next_state2', 'next_state3', 'reward'))


class Agent_v3:
    def __init__(self):
        self.brain = Brain_v3() 

    def update_q_function(self):
        return self.brain.replay()

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
        self.memory[self.index] = Transition(state[0], state[1], state[2], action, state_next[0], state_next[1], state_next[2], reward)
        self.index = (self.index + 1) % self.capacity 
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    def __len__(self):
        return len(self.memory)

class Net_v3(nn.Module):
    def __init__(self, n_in, n_mid, n_out):
        super(Net_v3, self).__init__()
        self.conv1 = nn.Conv2d(1, 1, 1) # (N_SERVER * N_JOB)
        self.conv2 = nn.Conv2d(1, 1, 1) # (N_SERVER * N_JOB)
        self.linear = nn.Linear(N_JOB, N_JOB) # (1 * N_JOB)

        self.fc1 = nn.Linear((2*N_SERVER+1)*N_JOB, n_mid)
        self.fc2 = nn.Linear(n_mid, n_mid)
        self.fc3 = nn.Linear(n_mid, n_out)

    def forward(self, x, y, z): # 输入state的三个信息
        x = F.leaky_relu(self.conv1(y)) # (N_SERVER * N_JOB)
        x = F.max_pool2d(x, 1) # (N_SERVER * N_JOB)
        y = F.leaky_relu(self.conv2(y)) # (N_SERVER * N_JOB)
        y = F.max_pool2d(y, 1) # (N_SERVER * N_JOB)
        z = F.leaky_relu(self.linear(z)) # (1 * N_JOB)
        # print(x.shape, y.shape, z.shape)
        # print(torch.cat((x[0],y[0]), 1).shape)
        k = torch.cat((x, y, z), 2)
        batch = k.shape[0]
        k = k.view(batch, (2*N_SERVER+1)*N_JOB)

        h1 = F.leaky_relu(self.fc1(k))
        h2 = F.leaky_relu(self.fc2(h1))
        output = self.fc3(h2)
        return output


class Brain_v3:
    def __init__(self):
        self.num_actions = N_SERVER 
        self.memory = ReplayMemory(CAPACITY)
        n_in, n_mid, n_out =  32, 32, N_SERVER
        if LOAD_OK1:
            self.main_q_network = torch.load(PATH1)
            self.target_q_network = torch.load(PATH1) 
        else:
            self.main_q_network = Net_v3(n_in, n_mid, n_out).to(device) 
            self.target_q_network = Net_v3(n_in, n_mid, n_out).to(device) 

        self.optimizer = optim.Adam(
            self.main_q_network.parameters(), lr=0.0001)
        # print(self.main_q_network)

    def replay(self):
        if len(self.memory) < BATCH_SIZE:
            return 0
        self.batch, self.state_batch, self.action_batch, self.reward_batch, self.non_final_next_states = self.make_minibatch()
        self.expected_state_action_values = self.get_expected_state_action_values()
        return self.update_main_q_network()

    def decide_action(self, state, episode):
        epsilon = 0.5 * (1 / (episode + 1))
        if epsilon <= np.random.uniform(0, 1):
        # if np.random.randint(1,10) >=2:
            self.main_q_network.eval() 
            with torch.no_grad():
                # print(state[0].shape, state[1].shape,state[2].shape)
                action = self.main_q_network(state[0], state[1], state[2]).max(1)[1].view(1, 1)
        else:
            action = torch.LongTensor(
                [[random.randrange(self.num_actions)]])
        return action

    def make_minibatch(self):
        transitions = self.memory.sample(BATCH_SIZE)
        batch = Transition(*zip(*transitions))
    
        state_batch1 = torch.cat(batch.state1)
        state_batch2 = torch.cat(batch.state2)
        state_batch3 = torch.cat(batch.state3)
        # print(batch.action)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        non_final_next_states1 = torch.cat([s for s in batch.next_state1
                                           if s is not None])
        non_final_next_states2 = torch.cat([s for s in batch.next_state2
                                            if s is not None])
        non_final_next_states3 = torch.cat([s for s in batch.next_state3
                                            if s is not None])

        non_final_next_states = [non_final_next_states1, non_final_next_states2, non_final_next_states3]
        state_batch = [state_batch1, state_batch2, state_batch3]
        return batch, state_batch, action_batch, reward_batch, non_final_next_states

    def get_expected_state_action_values(self):
        self.main_q_network.eval()
        self.target_q_network.eval()

        # self.state_action_values = self.main_q_network(self.state_batch).cuda(device).gather(1, self.action_batch)
        self.state_action_values = self.main_q_network(self.state_batch[0], self.state_batch[1], self.state_batch[2]).gather(1, self.action_batch.to(device)).to(device)

        non_final_mask = torch.ByteTensor(tuple(map(lambda s: s is not None, self.batch.next_state1)))
        next_state_values = torch.zeros(BATCH_SIZE).to(device)

        a_m = torch.zeros(BATCH_SIZE).type(torch.LongTensor).to(device)

        a_m[non_final_mask] = self.main_q_network(self.non_final_next_states[0], self.non_final_next_states[1], self.non_final_next_states[2]).detach().max(1)[1].to(device)
        a_m_non_final_next_states = a_m[non_final_mask].view(-1, 1).to(device)

        next_state_values[non_final_mask] = self.target_q_network(self.non_final_next_states[0], self.non_final_next_states[1], self.non_final_next_states[2]).gather(1, a_m_non_final_next_states).detach().squeeze()
        # print(self.reward_batch)
        # print(next_state_values)
        expected_state_action_values = self.reward_batch + GAMMA * next_state_values.to(device)
        return expected_state_action_values

    def update_main_q_network(self):
        self.main_q_network.train()
        loss = F.smooth_l1_loss(self.state_action_values,
                                self.expected_state_action_values.unsqueeze(1))
        # print("loss, ", loss)
        ret = copy(loss)
        self.optimizer.zero_grad()  
        loss.backward()  
        self.optimizer.step() 
        return ret

    def update_target_q_network(self): 
        self.target_q_network.load_state_dict(self.main_q_network.state_dict())
