import matplotlib.pyplot as plt

# 数据
curve1 = [18.14, 13.76, 12.26, 12.58, 13.05]   # PSO_TD3
curve2 = [16.02, 17.58, 15.40, 14.65, 15.48]   # PSO

# 横坐标（假设为迭代轮数或时间步）
x = range(1, len(curve1) + 1)

# 创建图形
plt.figure(figsize=(8, 5))

# 绘制两条曲线
plt.plot(x, curve1, marker='o', linestyle='-', color='blue', label='PSO_TD3')
plt.plot(x, curve2, marker='s', linestyle='--', color='red', label='PSO')

# 添加标签和图例
plt.xlabel('Iteration / Time Step')
plt.ylabel('Fitness Value')
plt.title('Fitness Convergence Comparison')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)

# 显示图形
plt.show()