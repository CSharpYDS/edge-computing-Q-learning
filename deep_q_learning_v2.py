from torch import FloatTensor
from brain import *
from env import *
from schedule_policy import *
# from brain_v2 import
def parseState(x,y,z):
    x = np.concatenate([x,y,z])
    x = x.reshape((1, (2*N_SERVER+1)*N_JOB))
    x = torch.from_numpy(np.array(x)).type(FloatTensor).to(device)
    return x

def deepQLearning_v2(job_sequence):
    agent = Agent1()
    cost_min, cost_min1 = 1000000, 1000000
    wrong = 0
    ret_history = []

    for episode in range(500):
        S = Servers()
        cost ,cost1 = 0, 0
        history = []
        idx = 0
        state, state_next = State(), State()
        tensor_state, tensor_state_next = None, None
        server_id = -1
        reward = 0

        for time in range(0, 100000):
            if idx >= len(job_sequence) and S.done(): 
                state_next.updateState(S, time)
                agent.memorize(tensor_state, action.to(device), tensor_state_next, reward)
                agent.update_q_function()
                break
            while True:
                if idx < len(job_sequence) and job_sequence[idx].depart_time == time:
                    job = job_sequence[idx]
                    idx+=1
                    one_hot = np.zeros((1, N_JOB))
                    one_hot[0][job.job_id] += 1
                    if server_id!=-1:
                        state_next_trans, state_next_queue, _ = state_next.updateState(S, time)
                        # tensor_state_next = torch.cat(torch.from_numpy(np.array([state_next_trans])).type(FloatTensor).to(device),
                        #                     torch.from_numpy(np.array([state_next_queue])).type(FloatTensor).to(device),
                        #                     torch.from_numpy(np.array([one_hot])).type(FloatTensor).to(device))
                        tensor_state_next = parseState(state_next_trans, state_next_queue, one_hot)
                        agent.memorize(tensor_state, action.to(device), tensor_state_next, reward)
                        agent.update_q_function()

                    # new round state
                    state_trans, state_queue, _ = state.updateState(S, time)
                    # tensor_state = [torch.from_numpy(np.array([state_trans])).type(FloatTensor).to(device),
                    #                 torch.from_numpy(np.array([state_queue])).type(FloatTensor).to(device),
                    #                 torch.from_numpy(np.array([one_hot])).type(FloatTensor).to(device)]
                    tensor_state = parseState(state_trans, state_queue, one_hot)
                    # 2.1 获取action
                    action = agent.get_action(tensor_state, episode)
                    server_id = action.cpu().item() # get a number
                    # 2.2 执行action
                    job.arrive_time = job.arriveTime(server_id)
                    S1 = Servers()
                    S1.clone(S)
                    reward = S1.getAddedCost(job, server_id, job.arrive_time)
                    S.servers[server_id].server.append(job)
                    reward = torch.from_numpy(np.array([-reward])).type(torch.FloatTensor).to(device)
                else:
                    break
            S, cost,cost1, history = SJFPolicy(S, time, cost, cost1,history) # 直接更新cost
        if(episode % 2 == 0):
            agent.update_target_q_function()

        cost_min = min(cost_min, cost)
        cost_min1 = min(cost_min1, cost1) 

        if judge(history) == False or len(history) != len(job_sequence):
            wrong += 1
            print(history)

    torch.save(agent.brain.main_q_network, PATH)
    print("wrong DQN", wrong)
    return cost_min, cost_min1, ret_history

