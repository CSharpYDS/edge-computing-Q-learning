import numpy as np
from env import *
from params import *
from schedule_policy import *

def updateQ(state, server_id, reward, new_state, Q, eta, gamma):
    s = state.count()
    a = server_id
    ns = new_state.count()
    Q[s, a] = Q[s, a] + eta * (reward + gamma * np.max(Q[ns, :]) - Q[s, a])
    return Q

def getAction(Q, state):
    s = state.count()
    return np.random.choice(np.where(Q[s] == np.max(Q[s]))[0])

def getMax(Q, state):
    s = state.count()
    return np.max(Q[s])

def QLearningPolicy(job_sequence):
    # 将state定义为当前系统中的job数量
    # state中包含每个server具有当前适合配置的个数
    # 当前server队伍的长度 + 系统中job的数量
    i = 0
    id = 0
    S = Servers()
    state = State()
    new_state = State()
    if Q_OK:
        Q = np.load('Q.npy')
    else:
        Q = np.zeros((MAX_JOB_SEQUENCE+1, N_SERVER))
    history = []
    cost_min = 1000000
    cost_min1 = 1000000
    wrong = 0
    ret_history = []
    for episode in range(300):
        S = Servers()
        cost = 0
        cost1 = 0
        history = []
        idx = 0
        id += 1
        for time in range(1, 100000):
            if idx >= len(job_sequence) and S.done():
                break
            job_in_time = []
            while True:
                if idx<len(job_sequence) and job_sequence[idx].depart_time == time:
                    job_in_time.append(job_sequence[idx])
                    idx+=1
                else:
                    break
            for job in job_in_time:
                state.updateState(S, time)
                # new action
                epsilon = 0.5 * (1 / (episode + 1))
                if epsilon <= np.random.uniform(0, 1):
                # if np.random.randint(1, 10) < 2:
                    server_id = np.random.randint(N_SERVER)
                else:
                    server_id = getAction(Q, state)
                # 执行action
                job.arrive_time = job.arriveTime(server_id)
                S.servers[server_id].server.append(job)
                # new state
                new_state.updateState(S, time+1)
                # reward
                reward = new_state.getCost()
                # Q-Learning
                Q = updateQ(state, server_id, -reward,
                            new_state, Q, eta, gamma)

            S, cost, cost1, history = SJFPolicy(S, time, cost, cost1, history)
        if cost < cost_min:
            ret_history = history
            cost_min = cost
            i = id
        cost_min1 = min(cost_min1, cost1)
        if judge(history) == False or len(history) != len(job_sequence):
            wrong += 1
    np.save('Q',Q)
    # print("wrong Q Learning", wrong)
    # print("Q Learning episode", i)
    return cost_min, cost_min1, ret_history
