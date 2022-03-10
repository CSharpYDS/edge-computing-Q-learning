#encoding: utf-8
import numpy as np
from params import *
from schedule_policy import *
from env import *

# 发送给最近的server
def nearestPolicy(job_sequence):
    servers = Servers()
    cost = 0
    cost1 = 0
    history = []
    i = 0
    for time in range(1, 100000):
        if i >= len(job_sequence) and servers.done():
            break
        for job in job_sequence:
            if job.depart_time == time:
                server_id = job.nearestServer()
                job.arrive_time = job.arriveTime(server_id)
                servers.add(server_id, job)
                i += 1
        servers, cost, cost1, history = SJFPolicy(servers, time, cost,cost1, history)
    return cost, cost1, history

def randomPolicy(job_sequence):
    cost_min, cost_min1 = 10000,10000
    for episode in range(2000):
        history = []
        cost, cost1 = 0, 0
        servers = Servers()
        i = 0
        for time in range(100000):
            if i >= len(job_sequence) and servers.done() : break

            for job in job_sequence:
                if job.depart_time == time:
                    server_id = np.random.randint(N_SERVER)
                    job.arrive_time = job.arriveTime(server_id)
                    servers.add(server_id, job)
                    i += 1

            servers, cost, cost1, history = SJFPolicy(servers, time, cost, cost1, history)

        if cost_min > cost and cost_min1 > cost1:
            cost_min = cost
            cost_min1 = cost1
    return cost_min, cost_min1, history
