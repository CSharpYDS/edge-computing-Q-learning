#encoding: utf-8
import numpy as np
from params import *

# schedule policy 在edge server上, 队列的调度算法

def FIFOPolicy(servers, cost, cost1, history):
    for i in range(len(servers.servers)):
        servers.servers[i].server = sorted(servers.servers[i].server ,key=lambda y:y.arrive_time)
        queue_time = 0
        for job in servers.servers[i].server:
            if job.arrive_time >= queue_time:
                finish_time = job.arrive_time+job.compute_time-1
                cost += float(finish_time - job.depart_time)/job.compute_time
                cost1 += finish_time
                queue_time = finish_time+1
                # arrive_time = depart_time + transmission_time
                # finish_time = arrive_time + compute_time - 1
                history.append((i, job.job_id, finish_time))
                # print(i, job.job_id,job.depart_time,job.compute_time,job.trans_time,queue_time-1)
            else:
                finish_time = queue_time+job.compute_time-1
                cost += float(finish_time - job.depart_time)/float(job.compute_time)
                cost1 += finish_time
                queue_time = finish_time+1
                history.append((i, job.job_id, finish_time))
                # print(i, job.job_id, job.depart_time,job.compute_time, job.trans_time,queue_time-1)
    return cost, cost1, history


def SJFPolicy(servers, time, cost, cost1, history, server_state): 
    for i in range(len(servers.servers)):
        x = servers.servers[i]
        if x.computing == None: # 没有正在计算的
            if len(x.server) == 0:
                continue
            x.sort(time)
            if len(x.waiting) != 0:
                x.addComputing(x.waiting[0],time) # 加入一个正在计算的
        # else: # 有正在计算的
        if x.computing != None:
            if x.computing_time <= time: # 已经计算完毕
                # server id, job id, finish_time) 
                pre = x.computing_time - x.computing.compute_time + 1
                history.append((i, x.computing.job_id, pre, x.computing.compute_time, x.computing_time))
                cost += float(x.computing_time - x.computing.depart_time)/float(x.computing.compute_time)
                cost1 += x.computing_time
                x.computing = None
                x.computing_time = 0
        servers.servers[i] = x
        if server_state != None:
            # print(server_state)
            # print(server_state[0])
            server_state[i][0].append(len(x.waiting))
            server_state[i][1].append(time)
    return servers, cost, cost1, history, server_state