#encoding: utf-8
import numpy as np
from baseline_policy import *
from params import *
from env import *
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import time
from deep_q_learning_v3 import *
from deep_q_learning_v2 import *
from deep_q_learning_v1 import *
from q_learning_v1 import *
from q_learning_v2 import *

def main_without_config():
    time1 = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime()) 
    improv = [[] for i in range(200)]
    improv1 = [[] for i in range(200)]

    for episode in range(10):
        k,kk,k1 = [],[],[]
        map = [ 0 for i in range(1000)]
        temp = [ None for i in range(1000)]
        job_sequences = []
        i = 0
        while True:
            job_sequence = generate_job_sequence()
            my_length = len(job_sequence)
            if map[my_length] > 0:
                continue
            map[my_length] += 1
            i += 1
            temp[my_length] = job_sequence
            if i > 30:
                break
        for i in range(1000):
            if temp[i] != None:
                job_sequences.append(temp[i])

        for job_sequence in job_sequences:
            my_length = len(job_sequence)
            print("length", len(job_sequence))
            cost_random_sjf,cost1, history_random_sjf = randomPolicy(job_sequence)
            cost_near_sjf,cost2, history_near_sjf = nearestPolicy(job_sequence)
            print("random+SJFPolicy",cost_random_sjf, cost1)
            print(history_random_sjf)
            print("nearest+SJFPolicy",cost_near_sjf, cost2)
            # cost_Q_sjf, cost3,  history_Q_sjf = QLearningPolicy1(job_sequence)
            # print("QLearning1", cost_Q_sjf, cost3)
            cost_DQN_sjf, cost4, history_DQN_sjf, _ = eval_v2(job_sequence)
            print("DQN", cost_DQN_sjf, cost4)
            print(history_DQN_sjf)
            
            improv[my_length].append(float(100.0*(cost_DQN_sjf - cost_random_sjf)/cost_random_sjf))
            improv1[my_length].append(float(100.0*(cost4 - cost1)/cost1))

    x = []
    y = []
    yy = []
    print(improv)
    for i in range(100):
        if len(improv[i]) > 0:
            y.append(float(1.0*sum(improv[i])/len(improv[i])))
            yy.append(float(1.0*sum(improv1[i])/len(improv1[i])))
            x.append(i)
    plt.figure()
    plt.subplot(121)  
    plt.plot(x,y,'y', label='(DQN - random)/random')
    plt.grid(axis='x')
    plt.ylabel('turn-around time imporvement(%)')
    plt.xlabel('job sequence length')
    plt.legend()
    plt.subplot(122)
    plt.plot(x,yy,'y', label='(DQN - random)/random')
    plt.savefig('result/' + time1+"--"+str(episode)+ str(".png"))
    plt.grid(axis='x')
    plt.ylabel('sum of JCT imporvement(%)')
    plt.xlabel('job sequence length')
    plt.legend()
    # x_major_locator=MultipleLocator(1)
    # y_major_locator=MultipleLocator(10)

    # ax=plt.gca()
    # ax.xaxis.set_major_locator(x_major_locator)
    # ax.yaxis.set_major_locator(y_major_locator)
    plt.savefig('result/'+time1+"--"+str(episode)+ str(".png"))

if __name__ == "__main__":
    main_without_config()
