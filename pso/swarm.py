import numpy as np

def pso_step(pos, vel, pbest, gbest, params, rng, v_max, start, goal,
             map_min=0.0, map_max=20.0, vel_noise=0.0):
    """
    向量化 PSO 单步更新，每个粒子使用各自的 (w, c1, c2)。
    :param pos, vel, pbest: (N, P, 2)
    :param gbest: (P, 2)
    :param params: (N, 3) 每个粒子的 w, c1, c2
    :param rng: np.random.Generator，只用于 r1、r2 和速度噪声；
                不同参数控制方法用同一个种子时，这部分随机序列完全相同
    """
    w = params[:, 0, None, None]
    c1 = params[:, 1, None, None]
    c2 = params[:, 2, None, None]

    r1 = rng.random(pos.shape)
    r2 = rng.random(pos.shape)
    noise = rng.standard_normal(pos.shape)

    vel = w * vel + c1 * r1 * (pbest - pos) + c2 * r2 * (gbest[None] - pos)
    if vel_noise > 0:
        vel = vel + vel_noise * noise
    vel = np.clip(vel, -v_max, v_max)

    pos = np.clip(pos + vel, map_min, map_max)
    pos[:, 0] = start
    pos[:, -1] = goal
    vel[:, 0] = 0.0
    vel[:, -1] = 0.0
    return pos, vel
