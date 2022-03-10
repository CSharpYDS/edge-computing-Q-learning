#encoding: utf-8
from params import *

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
        self.notwating = []
        self.computing_time = 0
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
    def add(self, id, job):
        self.servers[id].add(job)
    def done(self):
        for s in self.servers:
            if len(s.server) > 0 or s.computing!= None:
                return False
        return True    

# 随机生成工作序列
def generate_job_sequence(my_length):
    values = []
    length = np.random.randint(MIN_JOB_SEQUENCE, MAX_JOB_SEQUENCE) # 随机生成队列长度
    for i in range(my_length):
        job_id = np.random.randint(N_JOB)
        time = np.random.randint(MIN_TIMELINE,MAX_TIMELINE)
        values.append(Job(job_id, time))
    job_sequence = sorted(values,key=lambda x:x.depart_time)
    return job_sequence


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
            return False
        servers[server_id].update(arrive_time, -1)
        servers[server_id].update(finish_time, 1)

    return True


