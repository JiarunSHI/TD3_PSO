# 独立测试的在线 TD3 对照初始化

evaluate_model.py 中的 online_td3 对照，在每个地图/试验场景开始时新建随机初始化的
Actor、Critic1、Critic2。目标网络从各自新建网络复制权重；Adam 优化器和更新计数从零开始，
经验池为空。不会继承加载检查点中的 Actor/Critic 权重或已训练的优化器状态。

每个场景结束后丢弃该场景学习结果，下一个场景独立开始。前 5 个 PSO 步仅采集经验，
使用初始 Actor 加探索噪声；第 6 步起，经验足够时更新模型。

初始化种子由独立命名空间 204、测试 --seed、地图 seed 和 trial 派生，区别于探索与目标策略噪声种子。
comparison.csv / summary.json 的 online_td3 行记录 online_init_mode=fresh_random 与 online_init_seed。
相同测试场景及相同 --seed 可以复现权重，修改 --seed 可生成另一组初始化。

仅改变独立测试入口。训练过程中的验证仍使用训练启动时保留的初始 TD3 副本作为在线对照起点。
冻结 td3 使用加载的受训 Actor，冻结 initial_actor 使用已保存的初始 Actor；二者不更新。
现有模型文件无需重新训练，重新执行测试即可使用这一规则。

测试命令仍为：

```powershell
python -s -B evaluate_model.py --checkpoint <模型目录>/best_trained.pt --iters 50 --out results/test_fresh_online
```

输出目录需为空或不存在。如需逐场景对比此前测试，可使用 --cases 指向此前保存的场景目录。
