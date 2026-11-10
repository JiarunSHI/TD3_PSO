import numpy as np

MAP_SIZE = 10

START = np.array([17.5,15.50])
GOAL = np.array([2,3])

NUM_PARTICLES = 150
NUM_POINTS = 10

EPISODES=6
STEPS_PER_EP = 10

BATCH_SIZE = 256
UPDATE_STEPS = 2

GAMMA = 0.97
TAU = 0.005

COLLISION_PENALTY = 100

EARLY_STOP_K = 30
EARLY_STOP_EPS = 1e-3

# =========================
# 在线自适应 PSO（trainer/train.py）
# =========================
MAX_ITERS = 250
MAP_MIN, MAP_MAX = 0.0, 20.0
V_MAX = 2.0

# 动作 [-1, 1] 线性映射到的 PSO 参数范围：w, c1, c2（random 方法使用同一范围）
PARAM_LOW = (0.4, 1.0, 1.0)
PARAM_HIGH = (0.9, 2.0, 2.0)

FIXED_PARAMS = (0.7, 1.5, 1.5)
LDIW_RANGE = (0.9, 0.4)

# Four start-goal choices: the two diagonals and the horizontal/vertical midlines.
# Edge-midpoint obstacles have radius 0.85, so midline ends are inset to 1.2.
ENDPOINT_CHOICES = (
    ("diagonal_main", np.array([0.0, 0.0]), np.array([20.0, 20.0])),
    ("diagonal_anti", np.array([0.0, 20.0]), np.array([20.0, 0.0])),
    ("horizontal", np.array([1.2, 10.0]), np.array([18.8, 10.0])),
    ("vertical", np.array([10.0, 1.2]), np.array([10.0, 18.8])),
)
WARMUP_ITERS = 5
UPDATES_PER_ITER = 4
LR = 1e-3
EXPL_NOISE = (0.2, 0.05)
# 探索噪声的指数衰减时间常数，与单次实验 --iters 无关。
EXPL_DECAY_ITERS = 250.0

# 无硬裁剪的改善奖励；对数与偏移量用于数值尺度控制。
REWARD_PBEST = 5.0
REWARD_GBEST = 5.0
REWARD_COLLISION_PROGRESS = 2.0
REWARD_FIRST_FEASIBLE = 1.0
COLLISION_COST_OFFSET = 0.05
COLLISION_COST_HIT = 0.1
