# ViTFly Dynamics：动态森林中的视觉无人机避障

[原项目主页](https://www.anishbhattacharya.com/research/vitfly) &nbsp;
[原论文](https://arxiv.org/abs/2405.10391)

本项目基于 ICRA 2025 ViTFly，保留官方 ViT+LSTM 作为对比模型，并扩展为动态森林中的多帧行为克隆与统一 Benchmark。实验模型包括单帧、相邻双帧和隔一帧双帧三种输入。

## 目录

- [Installation](#installation)
- [Test (simulation)](#test-simulation)
  - [Ablation experiment](#ablation-experiment)
  - [Comparison experiment](#comparison-experiment)
- [Gather dataset](#gather-dataset)
- [Train](#train)
- [Citation](#citation)

## Installation

完整仿真环境以 **Ubuntu 20.04 + ROS Noetic** 为基准。只训练模型时可跳过 ROS、catkin、Flightmare 和 Unity，直接阅读 [Train](#train)。

### 1. 安装 ROS 与 catkin tools

安装 ROS Noetic 后执行：

```bash
source /opt/ros/noetic/setup.bash
sudo apt update
sudo apt install -y python3-catkin-tools
```

### 2. 创建工作空间并克隆代码

```bash
cd ~
mkdir -p catkin_ws/src
cd catkin_ws
catkin init
catkin config --extend /opt/ros/noetic
catkin config --merge-devel
catkin config --cmake-args \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_FLAGS=-fdiagnostics-color

cd ~/catkin_ws/src
git clone --branch code-release --single-branch \
  git@github.com:astringent-orange/vitfly-dynamics.git
cd vitfly-dynamics
```

### 3. 放置大型资产

#### 3.1 场景

下载 [`environments.tar.xz`](https://pan.baidu.com/s/14q8FcqlNW3rIg7haxARpHw?pwd=9527)（提取码：`9527`），从仓库根目录解压：

```bash
tar -xJf <path/to/environments.tar.xz> \
  -C flightmare/flightpy/configs/vision
```

#### 3.2 Unity渲染器

下载 [`flightrender.tar.xz`](https://pan.baidu.com/s/19u9rlteO3mBndggyONYz7A?pwd=9527)（提取码：`9527`），从仓库根目录解压：

```bash
tar -xJf <path/to/flightrender.tar.xz> \
  -C flightmare/flightrender
```

#### 3.3 预训练模型

下载 [`pretrained_models.tar.xz`](https://pan.baidu.com/s/13EyXOon9YnWpx5HPMuTMmw?pwd=9527)（提取码：`9527`），从仓库根目录解压：

```bash
tar -xJf <path/to/pretrained_models.tar.xz> -C models
```

#### 3.4 训练数据集

下载 [`dataset.tar.xz`](https://pan.baidu.com/s/17TVpN4KkvG-Y7gJA54uheg?pwd=9527)（提取码：`9527`），从仓库根目录解压：

```bash
mkdir -p training/datasets
tar -xJf <path/to/dataset.tar.xz> -C training/datasets
```

### 4. 安装依赖并构建

```bash
cd ~/catkin_ws/src/vitfly-dynamics
bash setup_ros.bash

cd ~/catkin_ws
catkin build
```

### 5. 创建 Python 环境

完整 ROS 仿真使用 Python 3.8。先安装与本机 CUDA 匹配的 PyTorch，再安装其余依赖：

```bash
conda create -n pubflight python=3.8 pip -y
conda activate pubflight
python -m pip install --upgrade pip

# CUDA 12.1 示例
pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

每次开始工作前执行：

```bash
conda activate pubflight
source ~/catkin_ws/devel/setup.bash
cd ~/catkin_ws/src/vitfly-dynamics
```


## Test (simulation)

正式测试按以下步骤执行。

### Common preparation

#### 1. 确认森林场景并生成固定 manifest

`environments.tar.xz` 已包含当前森林Benchmark场景。下面的场景生成器会跳过已有目录，只在场景缺失时根据 `trees/environment_0..19` 补充生成；随后重新构建固定manifest：

```bash
python3 envtest/benchmark/generate_forest_scenes.py \
  --config envtest/benchmark/configs/forest_benchmark_v1.yaml

python3 envtest/benchmark/build_manifest.py \
  --config envtest/benchmark/configs/forest_benchmark_v1.yaml \
  --output envtest/benchmark/manifests
```

Manifest 是冻结的 CSV 测试清单，每行代表一个 rollout case，并记录场景、地图、phase seed、飞行速度、森林密度和动态障碍条件。不同模型复用相同 case，保证结果可复现且能够配对比较。

#### 2. 确认模型配置

模型名称和权重在 `envtest/benchmark/configs/forest_benchmark_v1.yaml` 中定义：

```text
single          -> models/current_frame/current_frame_vitlstm_000099.pth
adjacent        -> models/previous_frame/previous_frame_vitlstm_000099.pth
skip_one        -> models/second_previous_frame/second_previous_frame_vitlstm_000099.pth
original_vitfly -> models/ViTLSTM_model.pth
```

直接使用 `launch_evaluation.bash` 时，`offset=0/1/2` 会依次选择前三个默认 checkpoint；显式传入 `model_path=...` 可覆盖默认值。Benchmark runner 根据 policy 配置选择模型，不需要单独传 offset。

#### 3. 运行最简单的视觉测试

```bash
VITFLY_ENV_LEVEL=forest_benchmark_v1 \
VITFLY_ENV_FOLDER=map_010_density_medium_dynamic_speed_2mps \
bash launch_evaluation.bash 1 vision fixed_env offset=0
```

该命令在一张已生成的森林场景中运行一次单帧模型，结果写入 `evaluation.yaml`。将 `offset` 改为 `1` 或 `2` 可切换另外两种输入模型。

### Ablation experiment

该实验在 validation maps 上使用完全相同的 cases 比较单帧、相邻双帧和隔一帧双帧模型。采用单变量扫描：每次只改变一个因素，另外两个因素保持基准值。

| 扫描因素 | 取值 | 固定条件 |
|---|---|---|
| 动态障碍速度 | 1 / 2 / 3m/s | 无人机5m/s、100棵树 |
| 森林密度 | 50 / 100 / 150棵 | 无人机5m/s、动态障碍2m/s |
| 无人机速度 | 3 / 5 / 7m/s | 100棵树、动态障碍2m/s |

共同基准为“无人机5m/s、100棵树、动态障碍2m/s”。它只运行一次，并在三组扫描中复用，因此实际使用7个不重复条件：

```text
baseline
dynamic_speed_1mps
dynamic_speed_3mps
forest_density_low
forest_density_high
flight_speed_3
flight_speed_7
```

每个模型运行：

```text
7个条件 × 10张地图 × 2个phase seed = 140轮
```

三个模型合计 **420轮**。汇总指标包括：

- 成功率：到达终点且全程零碰撞的rollout比例。
- 碰撞率：至少发生一次碰撞的rollout比例。
- 飞行时间：只统计成功rollout，并报告均值和中位数。
- 成功率的95% bootstrap CI。
- 相同case下模型间的paired difference。

单帧模型：

```bash
python3 envtest/benchmark/run_benchmark.py \
  --policy single
```

相邻双帧模型：

```bash
python3 envtest/benchmark/run_benchmark.py \
  --policy adjacent
```

隔一帧双帧模型：

```bash
python3 envtest/benchmark/run_benchmark.py \
  --policy skip_one
```

`--config` 默认使用 `envtest/benchmark/configs/forest_benchmark_v1.yaml`，`--cases` 默认使用 `envtest/benchmark/manifests/ablation_validation_cases.csv`，需要测试其他配置或manifest时仍可显式覆盖。消融实验一次只允许一个 `--policy`。未指定 `--output` 时，结果自动写入带当前时间的目录，例如：

```text
results/ablation/single_20260717_151230/
results/ablation/adjacent_20260717_173510/
results/ablation/skip_one_20260717_195845/
```

三个模型全部完成后直接运行：

```bash
python3 envtest/benchmark/summarize_results.py
```

汇总脚本自动选择三个模型各自最新修改的 `results.csv`。汇总表格和图像固定写入 `results/ablation/table/`，再次运行会覆盖上一次的汇总结果。需要汇总指定批次时，仍可显式传入多个 `--results` 和一个 `--output`。

汇总后生成三张图：

```text
ablation_dynamic_speed.png
ablation_forest_density.png
ablation_flight_speed.png
```

每张图的横轴是对应因素的三个取值，三条曲线对应 `single`、`adjacent` 和 `skip_one`。图中从上到下依次为带95% bootstrap CI的成功率、碰撞率和成功rollout的中位飞行时间；没有成功rollout时，飞行时间点留空。

主要参数：

- `--config`：场景、模型和评价标准配置，默认使用森林Benchmark配置。
- `--cases`：本次实验使用的固定 case manifest，默认使用消融validation manifest。
- `--policy`：本次运行的模型；消融实验只能指定一个。
- `--output`：可选。消融实验默认使用 `results/ablation/<policy>_YYYYMMDD_HHMMSS/`；主对比实验仍需显式指定。
- `--resume`：跳过结果目录中已经完成的 `(policy_id, case_id)`；恢复中断实验时需同时传入原来的 `--output`。
- `--scenario <name>`：可选，只运行指定条件，例如 `dynamic_speed_1mps`。

汇总脚本不自动排序或选择模型。结合 `summary.csv`、`paired_model_differences.csv` 和三张图，人工决定用于主对比实验的模型。

### Comparison experiment

该实验在未参与消融的 test maps 上比较人工选出的最优模型、官方ViTFly、FastPlanner和EGO-Planner。仍采用动态障碍速度、森林密度和无人机速度三个单因素扫描，共用同一个baseline：

```text
7个不重复条件 × 10张test地图 × 2个phase seed = 140轮/模型
4个模型 × 140轮 = 560轮
```

每次命令只测试一个模型。主对比默认使用森林Benchmark配置和 `comparison_test_cases.csv`，结果自动写入 `results/comparison/<model>_YYYYMMDD_HHMMSS/`。

人工选择的最优模型直接通过 `--policy` 指定：

```bash
python3 envtest/benchmark/run_comparison.py --policy adjacent
```

当 `--policy` 为 `single`、`adjacent` 或 `skip_one` 时，表示人工选中的最优模型。结果中统一显示为 `best_ours`，同时在每行记录实际的 `source_policy_id`、frame offset和checkpoint hash。

官方ViTFly：

```bash
python3 envtest/benchmark/run_comparison.py --policy vitfly
```

规划器接入后分别运行：

```bash
python3 envtest/benchmark/run_comparison.py --policy fastplanner
```

```bash
python3 envtest/benchmark/run_comparison.py --policy egoplanner
```

恢复中断实验时，显式传入原来的结果目录：

```bash
python3 envtest/benchmark/run_comparison.py \
  --policy adjacent \
  --output results/comparison/best_ours_20260717_163000 \
  --resume
```

四个模型全部完成后运行：

```bash
python3 envtest/benchmark/summarize_comparison.py
```

汇总脚本自动选择四个模型各自最新修改的 `results.csv`，并检查每个模型是否完整包含同一组140个cases；发现中断或不配对的结果时会要求先恢复实验。验证通过后覆盖写入 `results/comparison/table/`：

```text
summary.csv
summary.json
paired_model_differences.csv
comparison_dynamic_speed.png
comparison_forest_density.png
comparison_flight_speed.png
```

每张图包含成功率及95% bootstrap CI、碰撞率、成功rollout中位飞行时间三个纵向子图，以及 `best_ours`、`vitfly`、`fastplanner`、`egoplanner` 四条曲线。需要汇总指定批次时，可以重复传入 `--results` 并用 `--output` 指定目录。

#### FastPlanner/EGO-Planner接入接口

当前配置已保留 `fastplanner_ros` 和 `egoplanner_ros`，但默认 `enabled: false`。未接入时运行对应命令会在启动仿真前明确报错，不会生成无效实验结果。

两个规划器必须满足相同的生命周期接口：

```text
validate(policy_config)
start(case_config)
wait_ready(timeout)
stop()
metadata()
```

规划器可以在adapter内部订阅自己的里程计、深度图、点云或地图消息，也可以输出原生轨迹；但送入仿真控制器前必须转换为统一命令：

```text
topic: /kingfisher/dodgeros_pilot/feedthrough_command
type: dodgeros_msgs/Command
mode: 2
velocity frame: world
velocity: [vx, vy, vz]
```

接入步骤：

1. 下载并在同一catkin工作空间编译规划器及其依赖。
2. 编写ROS bridge，向规划器提供当前case的地图、目标点、期望速度和状态输入。
3. 在bridge中将规划器轨迹或控制量转换为上述world-frame LINVEL命令。
4. 在 `forest_benchmark_v1.yaml` 中填写 `launch_command`、`ready_topic`，并将 `enabled` 改为 `true`。
5. 在 `policy_adapters.py` 中实现对应规划器的启动、ready等待、进程监控和停止逻辑。
6. 先验证进程退出会记录为 `controller_error`，再完成单case测试，最后运行140轮正式测试。

所有规划器使用同一独立evaluator，指标仍为成功率、碰撞率、成功飞行时间、95%成功率CI、paired model difference和结构化失败原因。规划器内部报告的“成功”不能替代Benchmark evaluator的判定。

## Gather dataset

数据采集使用可访问完整状态的动态 A* expert，训练输入仍为深度图及对应监督标签。

### 1. 生成动态采集环境

`environments.tar.xz` 已包含当前 `dynamic_astar_medium/environment_0..100`，正常采集时可以跳过本步骤。只有场景缺失或需要重新生成时，才根据 `spheres_medium/environment_0..100` 执行：

```bash
python3 envtest/ros/generate_dynamic_astar_env.py \
  --source-level spheres_medium \
  --env-ids 0-100 \
  --num-dynamic 8 \
  --difficulty medium \
  --seed 10
```

生成器不会覆盖已有目录。如需重新生成，确认旧场景不再需要后增加 `--overwrite`。

### 2. 采集 expert 轨迹

```bash
bash launch_evaluation.bash 10 state
```

`10` 是本批次轨迹数，可按需要修改。采集结果写入 `envtest/ros/train_set/`。每批结束后脚本会自动删除 rejected 轨迹，并更新 `collection_summary.json`。

### Reject 轨迹的判定

每个 state rollout 结束后会自动执行筛选；不需要再手动运行校验脚本。以下任一情况都会使该轨迹被删除：

- evaluator 未报告成功，或发生任意一次碰撞；
- 缺少 `data.csv`、深度图、必需字段或环境字段，时间戳重复，深度图无效，或 CSV 行数与深度图数量不一致；
- A* 路径缓存缺失、整条轨迹没有成功的 A* 样本、轨迹越过终点后仍被记录，或实际速度指令出现反向分量；
- 发生碰撞但障碍物间距仍记录为非负值，或任一记录的最近障碍物间距小于 `0 m`；
- 实际沿 A* 路径连续倒退超过 `0.75 m`，或偏离路径超过 `1.5 m`；
- 候选速度、制动、避障和减速诊断字段缺失、非有限或互相矛盾，例如速度不在 `[0, desired_vel]`、紧急停车标记不正确、连续超过 2 帧在存在安全候选速度时仍无故保持零速，或速度变化超过 `3.5 m/s²`。

采集器还会报告但不会删除的质量告警：低速帧占比超过 15%、沿路径反向速度帧占比超过 2%、连续倒退超过 `0.3 m` 或偏离路径超过 `0.8 m`。这些告警保留在终端输出中，便于后续人工检查。

`collection_summary.json` 仅记录累计的 accepted/rejected 数量，不参与训练。

## Train

训练脚本需要 NVIDIA GPU。

### 1. 准备数据集

`training/datasets/dataset/` 的每个直接子目录代表一条完整轨迹，并包含 `data.csv` 和对应的深度 PNG。目录中只保留通过校验的轨迹。

### 2. 训练三种模型

从仓库根目录执行：

```bash
# 单帧：[D_t]
CUDA_VISIBLE_DEVICES=0 python3 training/train.py --offset 0

# 相邻双帧：[D_{t-1}, D_t]
CUDA_VISIBLE_DEVICES=0 python3 training/train.py --offset 1

# 隔一帧双帧：[D_{t-2}, D_t]
CUDA_VISIBLE_DEVICES=0 python3 training/train.py --offset 2
```

训练参数默认读取 `training/config/train_vitlstm.txt`，输出写入 `training/logs/<run_name>/`。查看训练曲线：

```bash
tensorboard --logdir training/logs
```

### 3. 放置最终权重

训练完成后，将最终 checkpoint 和同一次运行的 `run_metadata.json` 分别复制到：

```text
offset=0 -> models/current_frame/
offset=1 -> models/previous_frame/
offset=2 -> models/second_previous_frame/
```

如 checkpoint 文件名与默认配置不同，同步修改 `envtest/benchmark/configs/forest_benchmark_v1.yaml`。

## Citation

如果使用了原始 ViTFly 方法，请引用：

```bibtex
@inproceedings{bhattacharya2025vision,
  title={Vision transformers for end-to-end vision-based quadrotor obstacle avoidance},
  author={Bhattacharya, Anish and Rao, Nishanth and Parikh, Dhruv and Kunapuli, Pratik and Wu, Yuwei and Tao, Yuezhan and Matni, Nikolai and Kumar, Vijay},
  booktitle={2025 IEEE International Conference on Robotics and Automation (ICRA)},
  year={2025},
  organization={IEEE}
}
```

## Acknowledgements

仿真启动代码以及仓库中的 Flightmare、DodgeDrone Simulation 基础来自 [ICRA 2022 DodgeDrone Competition](https://github.com/uzh-rpg/agile_flight)。
