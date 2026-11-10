# 多回合训练与短迭代测试

> 版本说明：本文保留首次 9 维实验的设置与解释。当前版本为 11 维，默认 gamma=0.97、个体最优奖励系数=5；新命令、可行路径记录和初始模型选模规则见 [long_return_update.md](long_return_update.md)。

本文件对应 `train_model.py` 和 `evaluate_model.py`，取代旧的 23 维在线训练方案。
`main_pso.py` / `compare.py` 仍是旧的单次在线实验入口，本次训练请使用新入口。

## 首次训练命令

PowerShell，在已安装 NumPy 和 PyTorch 的环境中执行：

```powershell
Set-Location 'E:\BaiduNetdiskDownload\pso_td3'
python -s -B train_model.py --episodes 100 --train-iters 100 --eval-iters 50 --eval-every 10 --out results/pretrain_100
```

`-s` 禁用用户级 site-packages，避免本机用户目录 NumPy 与 Anaconda 依赖混用。
配置检查可加 `--dry-run`：不会运行训练、生成种群或写入结果。
结果目录必须为空或不存在；再次实验请更换 `--out`。

训练结束后独立测试：

```powershell
python -s -B evaluate_model.py --checkpoint results/pretrain_100/best.pt --iters 50 --out results/test_50
```

检查更短的 25 步预算时复用同一组已保存测试条件：

```powershell
python -s -B evaluate_model.py --checkpoint results/pretrain_100/best.pt --iters 25 --cases results/test_50/cases --out results/test_25
```

## 初始种群与随机性

- 默认每回合 150 个粒子，每条路径 10 个点，起终点固定。
- 每回合新建不传 seed 的 `np.random.default_rng()`，从系统熵初始化。
- **每个候选粒子**独立抽取一个 `8×2` 标准高斯增量数组；不是整个回合共用一个数组。
- 沿路径点轴累加增量，除以该累计数组的标准差加 `1e-8`，乘以 2.5，再加到起终点连线的 8 个内部点上。
- 越界就丢弃整条候选，继续抽取直到收齐 150 条。不因碰撞而拒绝路径。
- 这里高斯分布描述增量的抽样规律；归一化、累加、拒绝采样后的路径不再是独立高斯点。
- `--seed` 只控制网络、PSO 系数、探索、经验采样及地图抽取顺序，**不控制初始种群**。
- 相同命令再次启动不会复现相同初始种群，因此保存每回合的实际种群、地图和 PSO seed；测试也保存实际数组，供方法间公平共用。
- 原 `run_pso(pop_seed=...)` 保留给旧脚本及确定性回归测试，新训练入口不接受也不传入 `pop_seed`。
- 地图生成排除覆盖固定起终点的障碍物；这会改变部分旧地图种子对应的地图。结果中的实际障碍物数组是复核依据。

## 9 维状态

顺序为：`f-pbest_f`、`f-gbest_f`、`f`、`cos(v,pbest-x)`、`cos(v,gbest-x)`、
`cos(v,v_prev)`、最坏碰撞深度 H、碰撞标记 C、速度大小。

方向计算只使用可移动点：将 `8×2` 速度与吸引向量各自展平为 16 维向量后计算余弦。
任意一个向量为零时，余弦约定为 0。余弦限到 [-1,1] 只修正浮点舍入。

速度大小使用 `q = sqrt(mean(v**2))/v_max`，输入为 `2q-1`。
当前 `v_max=2` 是**每个速度坐标分量**的限幅，因此 q 在 [0,1] 内。
这里 v 指 PSO 保存的速度；位置触及地图边缘时，它可能不同于实际位置位移。

适应度三项使用相同的固定对数归一化：

```
F0 = 0.5 * ||goal-start||
Fmax = 0.5*(P-1)*sqrt(2)*(map_max-map_min) + 0.2*(P-1)*pi + 40*r_max
N(u) = 2*log(1+u/F0)/log(1+Fmax/F0) - 1
```

Fmax 是当前路径点位于地图内时的保守上界。H 输入为 `2H/r_max-1`；C 原始定义是
0/1，输入编码为 -1/+1。无障碍物时 H=0，深度参考尺度取 1。
训练和推理共用该变换，不使用随 batch/训练过程改变的均值方差，不硬裁剪适应度异常值。
增加速度大小能减少状态混淆；该压缩观测仍省略完整几何及首次可行历史，不能称为严格马尔可夫状态。

## 适应度与奖励

令 d(segment, center) 为圆心到有限线段的最短距离：

```
H = max_{segment, obstacle} max(radius - d(segment, center), 0)
f = 0.5*路径长度 + 0.2*总转角 + 0.2*最大转角 + 100*越界距离平方和 + 40*H
```

线段与障碍物相切也记碰撞，此时 H 可以为 0。因此 H=0 不等价于已经可行。
有限碰撞惩罚不能保证最低适应度路径无碰撞，日志同时报告 `best_collide` 和可行比例。
PSO 原有的地图位置裁剪保留，正常迭代中越界适应度项为 0；空间边界不等于回合终止边界。

从 t 到 t+1 的奖励：

```
r_pbest = 10 * max(log(pbest_f_t / pbest_f_next), 0)
r_adjacent_fitness = log(f_t / f_next)
r_gbest = 5 * max(log(gbest_f_t / f_next), 0)
r_adjacent_depth = 2 * (H_t-H_next) / r_max
r_first_feasible = 1[以前从未可行 且 当前不碰撞]
r = 上述五项之和
```

两项相邻改善均保留负值，奖励总值不设上限。初始已经可行的粒子不会领取首次可行奖励，
首次逃离后再次进出障碍物也不再领取该首次奖励。gbest 奖励以本步更新前的公共全局纪录为参照。

## 训练与截断

- 默认训练 100 回合，每回合 100 个 PSO 步，150 条 transition/步，共产生 150 万条经验。
- 模型、Adam 优化器、经验池、TD3 更新计数和探索步数跨回合持续；只重置种群、速度和路径纪录。
- 每回合重新初始化路径意味着一段新的采样轨迹，**不意味着环境到达终止状态**。
- 每条目标值都是 `r + gamma*min(Q1_target(s_next,a_next),Q2_target(s_next,a_next))`。
  包括最后第 100 步；s_next 是真实执行后状态，不是下一回合的初始化状态。
- 无 t/T；没有倒计时或终止 done 标记。默认 gamma=0.9，tau=0.005。
- 仅训练开始的前 20 个**全局** PSO 步执行随机动作，之后不在每个回合重复预热。
- 预热后，`sigma(g)=0.05+0.15*exp(-(g-20)/2000)`，g 是动作执行前已完成的全局 PSO 步数。
- 探索动作 `clip(actor(s)+Normal(0,sigma),-1,1)`；探索高斯本身不单独裁剪。
- Actor 为 9→128→128→3，隐藏层 ReLU，输出 tanh；Critic 两套独立网络，输入 9+3 维。
- Actor lr=1e-4，Critic lr=3e-4，batch=256，经验池容量 200000，每个 PSO 步更新 Critic 4 次，
  每 2 次 Critic 更新做一次 Actor 更新及目标网络软更新。经验不足时跳过梯度更新。
- 目标策略噪声标准差 0.2，噪声先限到 [-0.5,0.5]，叠加目标 Actor 后动作限到 [-1,1]。
- 动作线性映射：`w=0.65+0.25*a0`，`c1=1.5+0.5*a1`，`c2=1.5+0.5*a2`。
- 不添加抑制饱和的损失、logit 裁剪或权重投影。归一化并不保证 Actor 永不饱和；边界动作允许使用。

训练地图 seed 默认 0–19，每轮打乱后遍历。验证地图为 1000–1004，每张 3 个随机初始种群。
训练前评估一次未训练 Actor，之后每 10 回合及最后一回合评估；每次复用相同验证场景。
独立测试默认用 2000–2004，每张 3 个随机初始种群。验证/测试 50 步，不探索、不加经验、不更新模型。
对照包括固定参数 PSO (0.7,1.5,1.5)、LDIW (w从0.9降至0.4) 和随机参数 PSO。
各方法共享初始数组和 PSO 随机系数；LDIW 自身依赖迭代预算，这是它的基线定义。

## 输出解读

| 文件/字段 | 含义 |
|---|---|
| `episodes.csv` | 每回合训练结果、奖励分量、平均动作和饱和比例 |
| `steps.csv` | 每个 PSO 步的细粒度记录，包含探索标准差与 Critic loss |
| `validation.csv` | 每次验证中每张地图、试验、方法的结果 |
| 测试 `comparison.csv` | 独立冻结测试的逐场景结果 |
| `best_f` | 到迭代上限为止发现的历史最佳适应度，越小越好 |
| `best_iteration` | 最终这个 best_f 首次出现的步数；0 表示初始化就已发现 |
| `first_feasible_iteration` | 种群中首次出现任意可行粒子的步数；-1 表示该预算内没有 |
| `actor_boundary_ratio` | 无探索噪声的策略输出中，abs(a)>=0.98 的动作分量比例 |
| `exploration_clip_ratio` | 加噪后、裁剪前 abs(a)>1 的动作分量比例；预热随机动作时为 0 |
| `logit_gt3_ratio` | abs(pre-tanh)>3 的比例，辅助判断更深的 tanh 饱和 |
| `actor_boundary_w/c1/c2` | 每个参数各自接近边界的比例 |
| `exploration_clip_w/c1/c2` | 每个参数各自因噪声越界而裁剪的比例 |
| `best.pt` | 受训模型中，验证集平均 best_f 最小的推理模型 |
| `last.pt` | 最后一个已完成回合的推理模型 |
| `train_initials` | 每回合真实初始种群、地图、PSO seed、结果路径和曲线 |
| `validation_cases` / 测试 `cases` | 可重复使用的实际评估初始条件 |

比例按“粒子×动作分量×步数”统计；预热时也监测 Actor 自身输出，但当时执行的是随机动作。
best_iteration 表示首次发现该精度的时间，并非另一种独立的适应度指标；比较时先看同预算 best_f，
同时查看碰撞情况，再参考出现时间。不同质量的最优值不能仅按出现更早来判优。

模型保存状态/归一化/适应度/动作映射版本，旧 10/23 维模型不能直接加载。`.pt` 是推理检查点，
不含完整经验池及优化器状态，当前入口未提供断点续训功能。

## 验证方式

```powershell
python -s -B -m unittest discover -s test -p test_online_changes.py -v
python -s -B test/test_consistency.py
```

多回合集成测试禁用优化器，只验证调度、经验收集、文件输出及冻结推理；另有一次合成数据梯度测试，
检查 Bellman 目标没有因截断丢失未来价值。这些检查不等于完成正式训练或验证算法效果。
