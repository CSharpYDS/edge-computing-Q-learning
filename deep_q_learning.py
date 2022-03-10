from brain import *
from env import *
from schedule_policy import *

def deepQLearning(job_sequence):
    agent = Agent()
    S = Servers()
    history, ret_history = [], []
    cost_min, cost_min1 = 1000000, 1000000
    i, wrong = 0, 0

    for episode in range(300):
        S = Servers()
        cost, cost1, period_reward = 0, 0, 0
        history = []
        i, idx = i+1, 0
        state, state_next = State(), State()

        for time in range(1, 100000):
            if idx >= len(job_sequence) and S.done(): break
            job_in_time = []
            while True:
                if idx < len(job_sequence) and job_sequence[idx].depart_time == time:
                    job_in_time.append(job_sequence[idx])
                    idx+=1
                else: break
            for job in job_in_time:
                # 1. state
                _,_, state_server = state.updateState(S, time, job)
                tensor_state = torch.from_numpy(np.array([[state.count(), job.job_id]])).type(torch.FloatTensor)
                # 2.1 获取action
                action = agent.get_action(tensor_state, episode)
                server_id = action.item()
                # 2.2 执行action
                job.arrive_time = job.arriveTime(server_id)
                S.servers[server_id].server.append(job)
                # 4. state_next
                _, _, state_server_next  = state_next.updateState(S, time+1, None)
                tensor_state_next = torch.from_numpy(np.array([[state_next.count(), -1]])).type(torch.FloatTensor)
                # 3. reward
                reward = torch.from_numpy(np.array([-state_next.getCost()])).type(torch.FloatTensor)
                # 5. Q Learning
                # print(tensor_state, action, tensor_state_next, reward)
                agent.memorize(tensor_state, action, tensor_state_next, reward)
                agent.update_q_function()
            S, cost,cost1, history = SJFPolicy(S, time, cost, cost1,history) # 直接更新cost
        if(episode % 2 == 0):
            agent.update_target_q_function()
        cost_min = min(cost_min, cost)
        cost_min1 = min(cost_min1, cost1) 
        if judge(history) == False or len(history) != len(job_sequence):
            wrong += 1

    torch.save(agent.brain.main_q_network, PATH)
    # print("wrong DQN", wrong)
    return cost_min, cost_min1, ret_history

