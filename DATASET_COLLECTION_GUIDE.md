# 专家策略数据集采集与审核

## 采集命令

先加载 ROS 工作空间，并使用已经安装 ROS、NumPy、pandas 和 OpenCV 的 Python 环境：

```bash
cd /home/mian/UAV/Vitfly
source devel/setup.bash
cd src/vitfly
export VITFLY_PYTHON=/home/mian/miniconda3/envs/pubflight/bin/python3.8
```

正式采集使用 `state` 模式。该模式由当前动态 A* 专家产生速度标签，默认轮换 `environment_0..100`，并在每条 rollout 导航前重置动态障碍相位：

```bash
VITFLY_REAL_TIME_FACTOR=10 \
bash launch_evaluation.bash 101 state phase_seed_base=1606
```

`1606` 只是示例，下一批应使用没有与历史批次重叠的相位基数。若先做小批验证，可将 `101` 改为 `3` 或 `10`。不要同时运行两个采集进程，否则最新轨迹与 `evaluation.yaml` 的对应关系会丢失。

采集完成后，`launch_evaluation.bash` 会自动调用 `curate_dataset.py --apply`。不合格轨迹会从 `envtest/ros/train_set/` 删除，合格轨迹保留；累计统计写入 `collection_summary.json`。正式采集建议先使用 `VITFLY_REAL_TIME_FACTOR=10`，确认稳定后再尝试更高倍率。

## 训练集准入标准

当前自动筛选采用与专家策略审核器一致的严格标准。以下任一项失败，轨迹不会进入训练集：

- evaluator 成功完成，`number_crashes=0`，`is_collide=0`；
- `data.csv`、深度 PNG、时间戳完全对齐，时间戳无重复，根目录无 RGB debug 图；
- 所有必需规划字段和环境元信息存在且为有限值，A* 至少有成功样本；
- 没有终点后样本，即 `pos_x >= 60` 的行数为 0；
- `velcmd_x` 不为负，沿 A* 路径的负速度比例不超过 `2%`，连续路径回退不超过 `0.3m`；
- 路径横向误差不超过 `0.8m`，低速帧比例不超过 `15%`；
- 最近障碍物净空不小于 `0m`；
- 候选速度始终位于 `[0, desired_vel]`，紧急停车标记、让行恢复、减速诊断和制动字段一致，应用速度加速度不超过 `3.5m/s^2`；
- 存在正向安全候选时，候选让行策略保持零速不超过 2 帧。

## 采集后复核

自动筛选后，可再次运行全量验证：

```bash
python3 envtest/ros/validate_dataset.py envtest/ros/train_set \
  --require-env-fields \
  --require-multiple-envs \
  --max-low-speed-ratio 0.15 \
  --max-negative-path-speed-ratio 0.02 \
  --max-path-backtrack-distance 0.3 \
  --max-path-cross-track-error 0.8
```

只有该命令通过的轨迹才用于训练。若希望构造更保守的论文高质量子集，可额外使用 `--min-nearest-margin 0.3`，但这属于比当前专家准入标准更严格的筛选。

若需要保留失败案例用于分析，应在采集前备份 `train_set/` 和对应的 `evaluation.yaml`；默认流程会删除 rejected 轨迹，以保证训练目录只包含通过审核的数据。
