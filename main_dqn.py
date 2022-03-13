#encoding: utf-8
import numpy as np
from baseline_policy import *
from params import *
from env import *
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import time
from deep_q_learning_v3 import *
from tqdm import tqdm

def main_without_config():
    time1 = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime()) 
    for epoch in range(20): # 绘制训练的收敛图
        loss_arr = []
        loss_arr_temp = []
        loss_idx = []
        for episode in range(32): # 每次32个job sequence
            for i in (range(1,2)):
                job_sequence = generate_job_sequence()
                print("")
                print("length", len(job_sequence))
                cost_random_sjf,cost1, history_random_sjf = randomPolicy(job_sequence)
                print("r ", cost_random_sjf, cost1)
                cost_DQN_sjf, cost4, history_DQN_sjf, loss = deepQLearning_v3(job_sequence, None)
                print("q ", cost_DQN_sjf, cost4)
                loss_arr_temp.append(loss)
        for i in range(1200):
            loss_idx.append(i)
            loss_arr.append(0)
        for i in range(32):
            for j in range(1200):
                loss_arr[j] += loss_arr_temp[i][j]
            for j in range(1200):
                loss_arr[j] = float(1.0*loss_arr[j]/32)
        plt.figure()
        plt.plot(loss_idx, loss_arr, 'r', label='loss')
        plt.savefig('loss_result/loss1/' + time1+"--"+str(epoch)+ str(".png"))
        plt.close()
if __name__ == "__main__":
    main_without_config()
