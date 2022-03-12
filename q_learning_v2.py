import numpy as np
from env import *
from params import *
from schedule_policy import *
from copy import copy
def updateQ(state, server_id, reward, new_state, Q, eta, gamma, done):
    # s = state.count()
    a = server_id
    # ns = new_state.count()
    s = state
    ns = new_state
    # print(s, a, ns)
    if done == True:
        Q[s, a] = Q[s, a] + eta * (reward - Q[s, a])
    else:
        Q[s, a] = Q[s, a] + eta * (reward + gamma * np.max(Q[ns, :]) - Q[s, a])
    return Q

def getAction(Q, state):
    s = state.count()
    return np.random.choice(np.where(Q[s] == np.max(Q[s]))[0])

def getMax(Q, state):
    s = state.count()
    return np.max(Q[s])

def QLearningPolicy1(job_sequence):
    # 将state定义为当前系统中的job数量
    # state中包含每个server具有当前适合配置的个数
    # 当前server队伍的长度 + 系统中job的数量

    if Q_OK:
        Q = np.load('Q.npy')
    else:
        Q = np.zeros((MAX_JOB_SEQUENCE * (N_JOB+1) + 1, N_SERVER)) # 仍然是以job的数量作为系统状态

    cost_min, cost_min1 = 1000000, 1000000
    wrong = 0
    ret_history = []

    for episode in range(100):
        S = Servers()
        cost ,cost1 = 0, 0
        history = []
        idx = 0
        state, state_next = State(), State()
        state1, state2 = 0, 0
        server_id = -1
        reward = 0
        for time in range(1, 100000):
            if idx >= len(job_sequence) and S.done():
                state_next.updateState(S, time)
                state2 = state_next.count() * 0
                Q = updateQ(state1, server_id, reward, state2, Q, eta, gamma, done=True)
                break
            while True:
                if idx<len(job_sequence) and job_sequence[idx].depart_time == time:
                    job = job_sequence[idx]
                    idx+=1
                    # previous round
                    if server_id != -1:
                        state_next.updateState(S, time)
                        state2 = state_next.count() * (job.job_id + 1)
                        Q = updateQ(state1, server_id, reward, state2, Q, eta, gamma, done=False)

                    # new round
                    # new state
                    state.updateState(S, time)
                    state1 = state.count() * (job.job_id+1)
                    # new action
                    epsilon = 0.5 * (1 / (episode + 1))
                    if epsilon <= np.random.uniform(0, 1):
                        server_id = np.random.randint(N_SERVER)
                    else:
                        server_id = getAction(Q, state)
                    # new reward
                    job.arrive_time = job.arriveTime(server_id)
                    S1 = Servers()
                    S1.clone(S)
                    # print(S, S1)
                    reward = S1.getAddedCost(job, server_id, job.arrive_time)
                    # 执行action
                    S.servers[server_id].server.append(job)
                else:
                    break
            S, cost, cost1, history = SJFPolicy(S, time, cost, cost1, history)
        if cost < cost_min:
            ret_history = history
            cost_min = cost
        cost_min1 = min(cost_min1, cost1)

        if judge(history) == False or len(history) != len(job_sequence):
            wrong += 1
            # print(len(history), len(job_sequence))
            # print(history, job_sequence)
    np.save('Q',Q)
    print("wrong Q Learning", wrong)
    # print("Q Learning episode", i)
    return cost_min, cost_min1, ret_history
