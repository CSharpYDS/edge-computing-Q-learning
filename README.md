# edge-computing-Q-learning

trying Q learning and DQN


## dispatch policy baseline
- random
- nearest-first

## schedule policy baseline
- shortest job first(SJF)
- first in first out(FIFO)

## 分布
- 任务的计算时间符合泊松分布, lambda = 5 
- 某一类任务在总时间内到达的个数符合泊松分布 lambda = 2
- 某一类任务到达的时间间隔符合泊松分布 lambda = 3
- 任务i到服务j的传输时间依赖于ij, 范围为(1, 3)


# DQN
## version 1
### state
`state: [state.count(), job_id]`

当前系统job的总数 + 此时要分配的job id

### reward

job分配后, 下一时刻系统内的job总数

---
## version 2
### state

- `state: [state.trans, state.queue, job_id]` 
当前系统中所有job传输到server的情况 + 当前所有job在server上排队的情况 + 此时要分配的job id

```python
state.trans.shape = N_SERVER * N_JOB
state.queue.shape = N_SERVER * N_JOB
job_id.shape = 1 * N_JOB (one-hot)
```
三个信息拼接成形状为 `(1 , (2 * N_SERVER + 1) * N_JOB)` 的向量, 输入线性层 

### reward
- 首先优化目标设为 : 总周转时间

- 周转时间 = (任务完成计算时间 - 任务开始分配时间) / 任务计算时间

由于采用shortest - job -first 的调度策略, 每当分配一个新的job到服务器上时, 队列中比这个job计算时间更长的job等待时间将会变长

增加的时间为 sum((new_job_compute_time)/(long_job_compute_time)

所以每分配一个job, 都可以算出它对总周转时间的贡献, 把这个贡献设置为当前action的reward

---
## version 3
### state
- 将`state.trans`, `state.queue` 分别输入两个二维卷积层
- 将`job_id`输入一个线性层
- 然后将它们进行拼接操作, 再输入线性层

### reward
同version2
---
# Q Learning
## version 1
### state
- `state: [state.count()]`
当前系统job的总数

### reward
job分配后, 下一时刻系统内的job总数

---
## version 2
### state
- `state : [state_next.count() * (job.job_id)]`
设置为当前系统的job总数 * job_id 

### reward
设置为action对总周转时间的贡献, 同DQN version 2
