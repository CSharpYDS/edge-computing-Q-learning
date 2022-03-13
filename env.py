#encoding: utf-8
from params import *
from schedule_policy import *
from copy import copy

class Job(object):
    def __init__(self,job_id, time):
        self.job_id = job_id
        self.compute_time = t_compute[job_id]        
        self.depart_time = time
        self.arrive_time = time
        self.trans_time = time

    def transTime(self,server_id): 
        self.trans_time = t_transmission[self.job_id][server_id]
        return t_transmission[self.job_id][server_id]

    def arriveTime(self,server_id): # 设置job到达server上的时间
        self.arrive_time = self.depart_time+self.transTime(server_id)
        return self.arrive_time

    def nearestServer(self):
        return np.argmin(t_transmission[self.job_id])

class State(object):
    def __init__(self):
        self.trans_state = np.zeros((N_SERVER,N_JOB),dtype=np.int32)
        self.queue_state = np.zeros((N_SERVER,N_JOB),dtype=np.int32)
        self.submit_job_state = np.zeros((1, N_SERVER), dtype=np.int32)
        self.server_state = np.zeros((1, N_SERVER), dtype = np.int32)
    
    def count(self):
        return np.sum(self.trans_state) + np.sum(self.queue_state)
    def getCost(self):
        _penalty = BETA * (np.sum(self.queue_state)>=LQ)
        return _penalty + self.count()

    def updateState(self, S, time, job=None):
        if job != None:
            self.submit_job_state = t_transmission[job.job_id]
        server_id = 0
        self.trans_state = np.zeros((N_SERVER,N_JOB),dtype=np.int32)
        self.queue_state = np.zeros((N_SERVER,N_JOB),dtype=np.int32)
        for x in S.servers:
            for job in x.server:
                self.server_state[0][server_id] += 1
                if job.arrive_time <= time:
                    self.queue_state[server_id, job.job_id] += 1
                else:
                    self.trans_state[server_id, job.job_id] += 1
            if x.computing != None and x.computing_time <= time:
                self.queue_state[server_id, x.computing.job_id] += 1
                self.server_state[0][server_id] += 1

            server_id += 1
            
        return self.trans_state, self.queue_state, self.server_state
    
        

class Server(object):
    def __init__(self):
        self.server = []
        self.computing = None
        self.waiting = []
        self.notwaiting = []
        self.computing_time = 0
    def clone(self, s):
        self.server = copy(s.server)
        self.computing = copy(s.computing)
        self.waiting = copy(s.waiting)
        self.notwaiting = copy(s.notwaiting)
        self.computing_time = copy(s.computing_time)
        return self
    def add(self, job):
        self.server.append(job)
    def addComputing(self, job:Job, time):
        self.computing = job # 正在执行的job
        self.computing_time = time + job.compute_time-1 # 更新一下job完成计算的时间
        self.server.remove(job) # 从waiting和not waiting中删除job
    def sort(self, time): # 需要的参数: server, computing, computing_time, time
        self.waiting = [] # 已到达的
        self.notwaiting = [] # 尚未到达的
        for job in self.server:
            if job.arrive_time > time:
                self.notwaiting.append(job)
            else:
                self.waiting.append(job)
        self.waiting = sorted(self.waiting, key=lambda x:(x.compute_time, x.arrive_time))
        self.notwaiting = sorted(self.notwaiting, key=lambda x:(x.arrive_time,x.compute_time))
        self.server = self.waiting + self.notwaiting

class Servers(object):
    def __init__(self):
        self.servers = [Server() for i in range(N_SERVER)]
    def clone(self, S):
        for i in range(len(self.servers)):
            self.servers[i].clone(S.servers[i])
        return self
        
    def add(self, id, job):
        self.servers[id].add(job)
    def done(self):
        for s in self.servers:
            if len(s.server) > 0 or s.computing!= None:
                return False
        return True    
    def getAddedCost(self, job, server_id, time):
        S1 = self
        S1, _, _, _, _ = SJFPolicy(S1, job.arrive_time, 0, 0, [], None)
        cost = 0.0
        prefix = 0
        for x in S1.servers[server_id].waiting:
            if x.compute_time > job.compute_time:
                cost += float(1.0 * job.compute_time/x.compute_time)
            elif x.compute_time <= job.compute_time:
                cost += float(1.0 * x.compute_time/job.compute_time)
                prefix += x.compute_time
        notwaiting = sorted(S1.servers[server_id].notwaiting, key=lambda x:(x.compute_time, x.arrive_time))
        for x in notwaiting:
            if x.compute_time >= job.compute_time:
                if x.arrive_time <= time + prefix:
                    cost += float(1.0 * job.compute_time / x.compute_time)
                elif time + prefix < x.arrive_time and x.arrive_time <= time+prefix + job.compute_time:
                    gap = time + prefix + job.compute_time - x.arrive_time
                    cost += float(1.0 * gap / x.compute_time)
            else:
                if x.arrive_time < time + prefix:
                    prefix += x.arrive_time
                    cost += float(1.0 * x.compute_time / job.compute_time)
                elif x.arrive_time < time + prefix + job.compute_time:
                    gap = time + prefix + job.compute_time - x.arrive_time
                    cost += float(1.0 * gap / x.compute_time)
        cost += float(1.0*(job.arrive_time - job.depart_time)/job.compute_time)
        return cost


class BinaryIndexTree:
    def __init__(self, array: list):
        self._array = [0] + array
        n = len(array)
        for i in range(1, n + 1):
            j = i + (i & -i)
            if j < n + 1:
                self._array[j] += self._array[i]
    def lowbit(self, x: int) -> int:
        return x & (-x)
    def update(self, idx: int, val: int):
        prev = self.query(idx, idx + 1)
        idx += 1
        val -= prev    # val 是要增加的值
        while idx < len(self._array):
            self._array[idx] += val
            idx += self.lowbit(idx)

    def query(self, begin: int, end: int) -> int:
        return self._query(end) - self._query(begin)
    def _query(self, idx: int) -> int:
        res = 0
        while idx > 0:
            res += self._array[idx]
            idx -= self.lowbit(idx)
        return res

def judge(history):
    array = [0 for i in range(300)]
    servers = [BinaryIndexTree(array) for i in range(N_SERVER)]

    for x in history:
        server_id = x[0]
        job_id = x[1]
        finish_time = x[2] # 22
        compute_time = t_compute[job_id] # 8
        arrive_time = finish_time - compute_time + 1 # 15
        if servers[server_id].query(0,finish_time) != 0:
            print(x[0], x[1], x[2], compute_time, arrive_time)
            return False
        servers[server_id].update(arrive_time, -1)
        servers[server_id].update(finish_time, 1)

    return True



# 随机生成工作序列
def generate_job_sequence(my_length = 0):
    values = []
    ptr = 0
    # print("a")
    for j in range(4):
        job_num = np.random.poisson(10, N_JOB) # job 个数符合泊松分布
        time_gap = np.random.poisson(6, job_num)
        per = np.random.permutation(10)
        print(per, time_gap)
        for i in range(job_num):
            values.append(Job(per[i], ptr + time_gap[i%10]))
            ptr += time_gap[i%10]
    # print("xx")

    job_sequence = sorted(values,key=lambda x:x.depart_time) # 按照到达时间排序
    # for x in job_sequence:
    #     print(x.job_id, x.depart_time)
    return job_sequence
