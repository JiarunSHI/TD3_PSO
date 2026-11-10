# 11 维状态与累计回报验证

本次在原 9 维状态末尾增加两个距离，不加入种群多样性、停滞信息或停滞恢复机制。
Actor 的普通 tanh、动作范围、速度限幅、最坏碰撞深度 H 和相邻奖励保留负值的规则继续使用。

## 状态与参数

可移动点数 M=8。两个新增特征分别为：

```
d_p = sqrt(sum((pbest-x)^2)/(2*M))
d_g = sqrt(sum((gbest-x)^2)/(2*M))
state_distance = 2*d/(MAP_MAX-MAP_MIN)-1
```

固定起终点不参与计算。相同方向但不同吸引距离可以产生不同状态；该压缩表示仍不是完整马尔可夫状态。
训练与推理共用确定性归一化，不增加随 batch 变化的标准化。

默认 gamma 从 0.9 改为 0.97，个体最优奖励系数从 10 改为 5。分别可用
`--gamma` 和 `--reward-pbest` 覆盖，便于分开检验影响。其他奖励公式不变。

## 长期回报体现在哪里

Actor 每个 PSO 步输出一次 (w,c1,c2)。这一步的影响经过位置、速度、pbest、gbest 传到后续状态，
而不是让这三个参数在接下来若干步保持不变。

Critic 的目标仍为：

```
y = r + gamma * min(Q1_target(s_next,a_next), Q2_target(s_next,a_next))
actor_loss = -mean(Q1(s,actor(s)))
```

当前是一步 bootstrap 的 TD3。一步 bootstrap 不等于只优化一步奖励：下一状态 Q 递推表示之后的累计回报。
没有添加有限回合终止奖励，没有 t/T，最后一个采样步仍使用真实 next_state bootstrap。

gamma=0.97 时，20 步与 50 步后的奖励折扣分别约为 0.544 和 0.218；原 gamma=0.9 时约为
0.122 和 0.005。更大的 gamma 提高远期收益权重，但不会保证 Critic 学准，也不会凭空产生新的搜索方向。

训练新增以下日志，每个 PSO 步对实际执行的梯度更新取均值：

| 字段 | 定义 |
|---|---|
| `td_reward_mean` | 从经验池抽取的当步奖励 r 的平均值 |
| `td_reward_abs_mean` | mean(abs(r)) |
| `td_bootstrap_mean` | gamma*min(Q_target) 的平均值 |
| `td_bootstrap_abs_mean` | mean(abs(gamma*min(Q_target))) |
| `td_target_mean` | r+gamma*min(Q_target) 的平均值 |
| `td_q1_mean`, `td_q2_mean` | 更新前两个 Critic 对采样 (s,a) 的平均预测 |
| `td_actor_loss` | 本步发生 Actor 更新时的损失均值 |

无梯度更新时这些列为空值 NaN；尚无 Actor 更新时 actor_loss 也为 NaN。
`td_target_mean` 应约等于 `td_reward_mean + td_bootstrap_mean`，容许浮点舍入。
未来价值项大并不证明模型有远见，还需检查实际回报和优化结果，防止把价值估计偏大误判为能力提升。

每个奖励分量同时记录有符号均值、`_abs_mean` 和 `_nonzero_ratio`，后者以 abs(component)>1e-12 为非零。
因此能区分全局奖励“发生少”与“发生时也很小”，不能仅用有符号均值推断学习贡献。

## 训练命令

保留上一轮结果，复用上一轮的实际验证种群和地图；训练种群仍每回合从系统熵重新抽取。

```powershell
Set-Location 'E:\BaiduNetdiskDownload\pso_td3'
python -s -B train_model.py --episodes 100 --train-iters 100 --eval-iters 50 --gamma 0.97 --reward-pbest 5 --validation-cases results/pretrain_100/validation_cases --out results/pretrain_100_long_return
```

`--validation-cases` 会检查地图划分、粒子数量、路径点数和试验数，复制数组到新结果目录，不改原始文件。
若不传此参数，则重新生成验证种群。相同 `--seed` 不保证相同训练初始种群；每回合实际数组仍保存。
若需区分距离、gamma、奖励系数各自作用，应做单因素、多次独立训练，而不能把组合修改的收益归于某一项。
`--dry-run` 只打印并检查配置，不训练；数据文件内容在实际读取验证场景时检查。

## 模型保存与基线

- `initial.pt`：未经梯度更新的初始模型。
- `best.pt`：把第 0 回合也纳入选择后的验证均值最佳模型。训练未超过初始模型时，episode 可以为 0。
- `best_trained.pt`：仅训练后各次验证中的最佳模型。
- `last.pt`：最后一个完成回合的模型。

所有模型携带初始 Actor 权重，并在验证时记录平均参数。旧 9/10/23 维模型不兼容新的 11 维输入；
原始结果和原模型不会被覆盖，当前新入口不会隐式填补旧模型缺少的特征。
模型文件是推理检查点，没有保存完整经验池和优化器状态，不支持精确断点续训。

评估比较七种方法：当前 Actor、初始 Actor、原固定参数、参数范围中点、LDIW、随机参数、
当前 Actor 验证期间平均输出对应的固定参数。最后一种方法称为 `fixed_actor_mean`。
在验证时，它用整组验证轨迹上的平均输出作为固定参数；这个对照有助于诊断状态变化带来的收益。
在独立测试时，固定参数必须读取模型保存的验证均值，不用测试数据重新选择参数。
`td3_win_ratio` 在 td3 自身条目下为 null；其他方法条目仍表示 TD3 战胜该方法的场景比例。

```powershell
python -s -B evaluate_model.py --checkpoint results/pretrain_100_long_return/best_trained.pt --iters 50 --cases results/pretrain_100_long_return/validation_cases --out results/long_return_validation
python -s -B evaluate_model.py --checkpoint results/pretrain_100_long_return/best_trained.pt --iters 50 --out results/long_return_test
```

默认仍评估 50 步，同时记录第 10、20、30、50 步的 best_f。指定 --iters 100 时还会记录第 100 步。
这些中间值都是本次运行的前缀；LDIW 的权重计划依赖总预算，100 步运行的第 20 步不等于单独 20 步的 LDIW。
`discounted_reward_prefix` 是实际轨迹前 T 步的平均折扣奖励和，不包含 T 步后的尾部，不作为终止回报训练。
不同 gamma 或奖励权重下，该数值不能直接用来比较路径优化优劣。

## 最佳可行路径

增加与原 fitness 最优记录独立的可行路径档案，字段：

- `best_feasible_f`、`best_feasible_iteration`、`best_feasible_pos`。
- `feasible_curve`：截至每一步的历史最佳可行适应度。
- `ever_feasible_ratio`：评估场景中曾经找到可行路径的比例。
- `mean_best_feasible_f`：仅在存在可行路径的场景上取均值，必须结合 ever_feasible_ratio 阅读。

从未出现可行解时，JSON/CSV 的 best_feasible_f 为 null/空，位置数组为空，曲线对应值为 inf。
保留原 best_f、best_collide 以及 PSO 的 pbest/gbest 更新规则；新档案不改变搜索轨迹。
碰撞判定继续使用圆心到线段距离 <= 半径，相切仍判碰撞，没有隐式放宽几何容差。

## 验证

```powershell
python -s -B -m unittest discover -s test -p test_online_changes.py -v
python -s -B test/test_consistency.py
```

集成测试禁止正式优化器更新，验证跨回合状态保留、初始模型选模、验证场景复用、固定均值基线和冻结测试。
另用一次合成数据梯度计算验证持续任务的 bootstrap 和日志算术。
