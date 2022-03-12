#encoding: utf-8
import numpy as np
from baseline_policy import *
from params import *
from env import *
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import time
from deep_q_learning_v3 import *


def main_without_config():
    for episode in range(4):
        i = MIN_JOB_SEQUENCE
        for i in range(MIN_JOB_SEQUENCE, MAX_JOB_SEQUENCE):
            # job sequence 
            my_length = i
            job_sequence = generate_job_sequence(my_length)
            print("")
            print("length", len(job_sequence))
            cost_random_sjf,cost1, history_random_sjf = randomPolicy(job_sequence)
            print("r ", cost_random_sjf, cost1)
            cost_DQN_sjf, cost4, history_DQN_sjf = deepQLearning_v3(job_sequence)
            print("q ", cost_DQN_sjf, cost4)
            

if __name__ == "__main__":
    main_without_config()
