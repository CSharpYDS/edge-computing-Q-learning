#encoding: utf-8
import numpy as np
from baseline_policy import *
from params import *
from env import *
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import time
from deep_q_learning import *
from q_learning import QLearningPolicy

def main_without_config():
    time1 = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime()) 
    for episode in range(1):
        x,xx,x1 = [],[],[]
        y,yy,y1 = [],[],[] 
        z,zz,z1 = [],[],[] 
        k,kk,k1 = [],[],[]
        i = MIN_JOB_SEQUENCE
        while True:
            # job sequence 
            my_length = i
            job_sequence = generate_job_sequence(my_length)
            print("")
            print("length", len(job_sequence))
            cost_random_sjf,cost1, history_random_sjf = randomPolicy(job_sequence)
            cost_near_sjf,cost2, history_near_sjf = nearestPolicy(job_sequence)
            
            print("random+SJFPolicy",cost_random_sjf, cost1)
            # print("nearest+SJFPolicy",cost_near_sjf, cost2, history_near_sjf)
            cost_Q_sjf, cost3,  history_Q_sjf = QLearningPolicy(job_sequence)
            print("QLearning", cost_Q_sjf, cost3)
            cost_DQN_sjf, cost4, history_DQN_sjf = deepQLearning(job_sequence)
            print("DQN", cost_DQN_sjf, cost4)
            
            x.append(cost_random_sjf)
            y.append(cost_near_sjf)
            z.append(cost_Q_sjf)
            k.append(cost_DQN_sjf)
            xx.append(cost1)
            yy.append(cost2)
            zz.append(cost3)
            kk.append(cost4)
            x1.append(my_length)
            y1.append(my_length)
            z1.append(my_length)
            k1.append(my_length)

            i += 1
            if i >= MAX_JOB_SEQUENCE: break

        # f, ax = plt.subplots(1,2)
        plt.figure()
        plt.subplot(221)  
        plt.plot(x1,x,'r', label='random')
        plt.plot(y1,y,'g', label='nearest')
        plt.plot(z1,z,'b', label='Q Learning')
        plt.plot(k1,k,'y', label='DQN')
        plt.grid(axis='x')
        plt.ylabel('sum of turn-around time')
        plt.xlabel('job sequence length')
        # x_major_locator=MultipleLocator(1)
        # y_major_locator=MultipleLocator(5)
        plt.legend()
        # ax=plt.gca()
        # ax.xaxis.set_major_locator(x_major_locator)
        # ax.yaxis.set_major_locator(y_major_locator)

        plt.savefig('result/' + time1+"--"+str(episode)+ str(".png"))

        plt.subplot(222)
        plt.plot(x1,xx,'r', label='random')
        plt.plot(y1,yy,'g', label='nearest')
        plt.plot(z1,zz,'b', label='Q Learning')
        plt.plot(k1,kk,'y', label='DQN')
        plt.savefig('result/' + time1+"--"+str(episode)+ str(".png"))
        plt.grid(axis='x')
        plt.ylabel('sum of JCT')
        plt.xlabel('job sequence length')
        plt.legend()
        # x_major_locator=MultipleLocator(1)
        # y_major_locator=MultipleLocator(10)

        # ax=plt.gca()
        # ax.xaxis.set_major_locator(x_major_locator)
        # ax.yaxis.set_major_locator(y_major_locator)
        plt.savefig('result/'+time1+"--"+str(episode)+ str(".png"))

        # np.save('x'+time1, [x,xx,x1])
        # np.save('y'+time1, [y,yy,y1])
        # np.save('z'+time1, [z,zz,z1])
        # np.save('k'+time1, [k,kk,k1])
if __name__ == "__main__":
    main_without_config()
