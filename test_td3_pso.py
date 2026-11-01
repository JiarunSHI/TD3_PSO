import numpy as np
import torch
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import START, GOAL, NUM_POINTS, NUM_PARTICLES   # 注意：使用 NUM_PARTICLES
from env.map_env import MapEnv
from env.fitness import fitness
from pso.particle import Particle
from pso.swarm import update_particle
from drl.td3 import TD3

# 绘图函数（假设已存在）
from utils.visualization import plot_map


def get_state(p, gbest_f, var, delta_f, delta_g):
    return np.array([
        p.f,
        p.pbest_f,
        gbest_f,
        delta_f,
        delta_g,
        p.f - p.pbest_f,
        var
    ], dtype=np.float32)


def main():
    # 1. 环境与障碍物
    map_env = MapEnv()
    obstacles = map_env.generate_mixed()

    # 2. 粒子群初始化（使用 NUM_PARTICLES）
    swarm = [Particle(NUM_POINTS, obstacles) for _ in range(NUM_PARTICLES)]
    for p in swarm:
        p.prev_f = p.f   # 初始化历史适应度

    # 3. 加载训练好的 TD3 模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    td3 = TD3(device, action_bound=1.0)
    model_path_prefix = "models/td3_final"
    if not os.path.exists(f"{model_path_prefix}_actor.pth"):
        print(f"错误：未找到模型文件 {model_path_prefix}_actor.pth")
        return
    td3.load_model(model_path_prefix)
    print("模型加载成功！")

    # 4. 推理参数：固定迭代 50 次
    MAX_ITER = 50

    # 5. 初始化全局最优
    gbest = min(swarm, key=lambda p: p.f)
    gbest_f = gbest.f
    gbest_pos = gbest.pos.copy()

    c_global = 0
    history = []   # 记录每次迭代的全局最优适应度

    print(f"开始推理，共 {MAX_ITER} 次迭代...\n")
    for step in range(MAX_ITER):
        # 计算种群方差
        var = np.var([p.f for p in swarm])
        old_g = gbest_f

        # 收集状态并获取动作
        actions = []
        for p in swarm:
            delta_f = p.prev_f - p.f
            s = get_state(p, gbest_f, var, delta_f, 0.0)
            s_tensor = torch.FloatTensor(s).unsqueeze(0).to(device)
            with torch.no_grad():
                a = td3.actor(s_tensor).detach().cpu().numpy()[0]
            actions.append(a)

        # 执行动作（更新粒子）
        for i, p in enumerate(swarm):
            update_particle(p, gbest_pos, actions[i])

        # 更新适应度 + 个体最优
        for p in swarm:
            p.prev_f = p.f
            p.f = fitness(p.pos, p.obstacles)
            if p.f < p.pbest_f:
                p.pbest = p.pos.copy()
                p.pbest_f = p.f
                p.stagnation = 0
            else:
                p.stagnation += 1

        # 更新全局最优
        new_best = min(swarm, key=lambda p: p.f)
        if new_best.f < gbest_f:
            gbest_f = new_best.f
            gbest_pos = new_best.pos.copy()
            c_global = 0
        else:
            c_global += 1

        history.append(gbest_f)

        # 输出进度（每10次或最后一次）
        if (step + 1) % 10 == 0 or step == MAX_ITER - 1:
            print(f"迭代 {step+1:3d} | 全局最优适应度: {gbest_f:.6f}")

    # 6. 结果输出与可视化
    print(f"\n最终最优适应度: {gbest_f:.6f}")
    print(f"最优路径点数量: {len(gbest_pos)}")

    # 使用 plot_map 绘制结果
    plot_map(obstacles, gbest_pos, START, GOAL)

    # 绘制适应度下降曲线
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(history)
    plt.xlabel('迭代次数')
    plt.ylabel('全局最优适应度')
    plt.title('收敛曲线')
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()