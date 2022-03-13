from torch import FloatTensor
from brain import *
from env import *
from schedule_policy import *
from brain_v3 import *
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
from tqdm import tqdm
import time 

def parseState(x,y,z):
    ret = [torch.from_numpy(np.array([[x]])).type(FloatTensor).to(device),
            torch.from_numpy(np.array([[y]])).type(FloatTensor).to(device),
            torch.from_numpy(np.array([[z]])).type(FloatTensor).to(device)]
    return ret

def deepQLearning_v3(job_sequence, time1 = None):
    agent = Agent_v3()
    cost_min, cost_min1 = 1000000, 1000000
    wrong = 0
    ret_history = []
    loss_arr = []
    loss_idx = []
    if time1 == None:
        time1 = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime()) 
    final_server_state = [[[],[]] for i in range(N_SERVER)]
    for episode in tqdm(range(1200)):
        S = Servers()
        cost ,cost1 = 0, 0
        history = []
        idx = 0
        state, state_next = State(), State()
        tensor_state, tensor_state_next = None, None
        server_id = -1
        reward = 0
        loss = 0
        server_state = [[[],[]] for i in range(N_SERVER)]
        
        for t in range(0, 100000):
            if idx >= len(job_sequence) and S.done(): 
                state_next.updateState(S, t)
                agent.memorize(tensor_state, action.to(device), tensor_state_next, reward)
                x = agent.update_q_function()
                loss = loss + x
                break
            while True:
                if idx < len(job_sequence) and job_sequence[idx].depart_time == t:
                    job = job_sequence[idx]
                    idx+=1
                    one_hot = np.zeros((1, N_JOB))
                    one_hot[0][job.job_id] += 1
                    if server_id!=-1:
                        state_next_trans, state_next_queue, _ = state_next.updateState(S, t)
                        tensor_state_next = parseState(state_next_trans, state_next_queue, one_hot)
                        agent.memorize(tensor_state, action.to(device), tensor_state_next, reward)
                        x = agent.update_q_function()
                        loss = loss + x

                    # new round state
                    state_trans, state_queue, _ = state.updateState(S, t)
                    tensor_state = parseState(state_trans, state_queue, one_hot)
                    # 2.1 获取action
                    action = agent.get_action(tensor_state, episode)
                    server_id = action.cpu().item() # get a number
                    action = action.cuda()
                    # 2.2 执行action
                    job.arrive_time = job.arriveTime(server_id)
                    S1 = Servers()
                    S1.clone(S)
                    reward = S1.getAddedCost(job, server_id, job.arrive_time)
                    S.servers[server_id].server.append(job)
                    reward = torch.from_numpy(np.array([-reward])).type(torch.FloatTensor).to(device)
                else:
                    break
            S, cost,cost1, history, server_state = SJFPolicy(S, t, cost, cost1,history, server_state) # 直接更新cost
        if(episode % 2 == 0):
            agent.update_target_q_function()

        if cost_min > cost and cost_min1 > cost1:
            ret_history = history
            final_server_state = server_state
        cost_min = min(cost_min, cost)
        cost_min1 = min(cost_min1, cost1) 
    
        loss_arr.append(loss.cpu().item())
        loss_idx.append(episode)
        if judge(history) == False or len(history) != len(job_sequence):
            wrong += 1
            print(history)
    plt.figure()
    plt.plot(loss_idx, loss_arr,'r', label='loss')
    plt.savefig('loss_result/temp/' + time1 + str(".png"))
    plt.close()

    # plt.show()
    plt.figure()
    plt.subplot(151)
    plt.plot(final_server_state[0][1], final_server_state[0][0],'r', label='loss')
    plt.subplot(152)
    plt.plot(final_server_state[1][1], final_server_state[1][0],'r', label='loss')
    plt.subplot(153)
    plt.plot(final_server_state[2][1], final_server_state[2][0],'r', label='loss')
    plt.subplot(154)
    plt.plot(final_server_state[3][1], final_server_state[3][0],'r', label='loss')
    plt.subplot(155)
    plt.plot(final_server_state[4][1], final_server_state[4][0],'r', label='loss')
    plt.savefig('loss_result/temp_server/' + time1 + str(".png"))
    plt.close()
    torch.save(agent.brain.main_q_network, PATH1)
    print("wrong DQN", wrong)
    return cost_min, cost_min1, ret_history, loss_arr

