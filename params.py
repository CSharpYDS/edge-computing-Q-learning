#encoding: utf-8
import numpy as np
PATH = '/Users/chen/Downloads/code/zxy_DQN_sum_state/model/model.pt'

LOAD_OK = True
V_OK = True
Q_OK = False

BETA = 100

# 服务器数量
N_SERVER = 5
# 配置数量
N_CONFIG = 20
# 函数数量
N_FUNC = 20
# 任务数量
N_JOB = 20
# 队列最大容量
LQ = 60

# 工作序列的时间线范围
MIN_TIMELINE = 1
MAX_TIMELINE = 10

# 生成的job sequence长度范围
MIN_JOB_SEQUENCE = 3
MAX_JOB_SEQUENCE = 40

# job的传输时间范围
MIN_TRANS_TIME = 1
MAX_TRANS_TIME = 3
# job的计算时间范围
MIN_COMPUTE_TIME = 2
MAX_COMPUTE_TIME = 10

# 每个config所需的数量范围
MIN_CONFIG = 1
MAX_CONFIG = 3

# Q Learning
gamma = 0.9
eta = 0.1

# 定数の設定
GAMMA = 0.9  # 時間割引率
NUM_EPISODES = 500  # 最大試行回数


# 生成随机参数
def randomTimes():
    # job传输到server上的时间 
    t_transmission = np.random.randint(MIN_TRANS_TIME,MAX_TRANS_TIME,(N_JOB,N_SERVER)) #TODO：设置范围 
    # job计算的时间
    t_compute = np.random.randint(MIN_COMPUTE_TIME,MAX_COMPUTE_TIME,N_JOB) #TODO：设置范围
    return t_transmission,t_compute

def randomConfig():
    # job与所需配置的对应关系
    configuration = np.random.random((N_JOB, N_CONFIG))
    temp = np.zeros(N_CONFIG)
    temp[:MAX_CONFIG] = 1 # 每个job只有3个需要的配置
    for i in range(N_JOB):
        np.random.shuffle(temp)
        configuration[i] = temp
    # t_local 本地运行，相当于任务本身的运行时间+排队时间, 不考虑传输成本
    # t_relay 转发到具有相应配置的Edge Server上计算, 相当于传输任务的成本
    t_relay = np.random.randint(MIN_TRANS_TIME,MAX_TRANS_TIME)
    # t_bypass 发送到Remote Cloud上进行计算
    t_bypass = t_relay * 10
    # t_fetch 从Remote Cloud中下载相应配置到指定服务器的成本, 由于下载时间>>传输时间,
    # 所以将忽略传输时间, 统一一个下载时间
    t_fetch = t_bypass * 10
    # TODO: job到server的对应关系
    return configuration,t_relay, t_bypass, t_fetch

# 赋予全局参数
if V_OK:
    configuration = np.load('data/configuration.npy')
    t_relay = np.load('data/t_relay.npy')
    t_bypass = np.load('data/t_bypass.npy')
    t_fetch = np.load('data/t_fetch.npy')
    t_transmission = np.load('data/t_transmission.npy')
    t_compute = np.load('data/t_compute.npy')

else:
    configuration, t_relay, t_bypass, t_fetch = randomConfig()
    t_transmission,t_compute = randomTimes()
    np.save('data/configuration', configuration)
    np.save('data/t_relay', t_relay)
    np.save('data/t_bypass', t_bypass)
    np.save('data/t_fetch', t_fetch)
    np.save('data/t_transmission', t_transmission)
    np.save('data/t_compute', t_compute)
