# ViTFly Dynamics：动态森林中的视觉无人机避障

[原项目主页](https://www.anishbhattacharya.com/research/vitfly) &nbsp;
[原论文](https://arxiv.org/abs/2405.10391)

本项目基于 ICRA 2025 ViTFly，保留官方 ViT+LSTM 作为对比模型，并扩展为动态森林中的多帧行为克隆与统一 Benchmark。实验模型包括单帧、相邻双帧和隔一帧双帧三种输入。

## 目录

- [Installation](#installation)
- [Test (simulation)](#test-simulation)
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

### 3. 放置大型资产（待补充）

> TODO：下载地址、文件名和校验和由维护者在发布资产后补充。

| 资产 | 目标位置 |
|---|---|
| Flightmare Unity renderer | `flightmare/flightrender/` |
| 原始 `trees` 场景 | `flightmare/flightpy/configs/vision/trees/` |
| 原始 `spheres_medium` 场景 | `flightmare/flightpy/configs/vision/spheres_medium/` |
| 官方 ViT+LSTM 权重 | `models/ViTLSTM_model.pth` |
| 单帧模型权重和 `run_metadata.json` | `models/current_frame/` |
| 相邻双帧模型权重和 `run_metadata.json` | `models/previous_frame/` |
| 隔一帧双帧模型权重和 `run_metadata.json` | `models/second_previous_frame/` |
| 训练数据集 | `training/datasets/dataset/` |

```bash
# TODO：将占位符替换为实际资产包路径
tar -xf <environment-archive.tar> -C flightmare/flightpy/configs/vision
tar -xf <renderer-archive.tar> -C flightmare/flightrender
unzip <dataset-archive.zip> -d training/datasets/dataset
```

新训练的 checkpoint 必须与同一次训练生成的 `run_metadata.json` 放在一起。官方权重通过 legacy adapter 加载，不需要该文件。

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

每次开始工作时执行：

```bash
conda activate pubflight
source ~/catkin_ws/devel/setup.bash
cd ~/catkin_ws/src/vitfly-dynamics
```

`devel/setup.bash` 已记录工作空间的 ROS Noetic underlay，因此会同时加载 `/opt/ros/noetic`，不需要在每个新终端重复执行两条 `source`。如果当前终端已经加载过工作空间，也不需要再次执行；只有尚未构建工作空间或无法找到 ROS 命令时，才单独执行 `source /opt/ros/noetic/setup.bash`。

从仓库根目录运行 `launch_evaluation.bash` 时，脚本会自动使用当前仓库中的 `flightmare/`，无需手动设置 `FLIGHTMARE_PATH`。

## Test (simulation)

正式测试按以下步骤执行。

### 1. 生成森林场景和固定 manifest

该步骤要求 `trees/environment_0..19` 已放置完成：

```bash
python3 envtest/benchmark/generate_forest_scenes.py \
  --config envtest/benchmark/configs/forest_benchmark_v1.yaml

python3 envtest/benchmark/build_manifest.py \
  --config envtest/benchmark/configs/forest_benchmark_v1.yaml \
  --output envtest/benchmark/manifests
```

Manifest 是冻结的 CSV 测试清单，每行代表一个 rollout case，并记录场景、地图、phase seed、飞行速度、森林密度和动态障碍条件。不同模型复用相同 case，保证结果可复现且能够配对比较。

### 2. 确认模型配置

模型名称和权重在 `envtest/benchmark/configs/forest_benchmark_v1.yaml` 中定义：

```text
single          -> models/current_frame/current_frame_vitlstm_000099.pth
adjacent        -> models/previous_frame/previous_frame_vitlstm_000099.pth
skip_one        -> models/second_previous_frame/second_previous_frame_vitlstm_000099.pth
original_vitfly -> models/ViTLSTM_model.pth
```

直接使用 `launch_evaluation.bash` 时，`offset=0/1/2` 会依次选择前三个默认 checkpoint；显式传入 `model_path=...` 可覆盖默认值。Benchmark runner 根据 policy 配置选择模型，不需要单独传 offset。

### 3. 运行最简单的视觉测试

```bash
VITFLY_ENV_LEVEL=forest_benchmark_v1 \
VITFLY_ENV_FOLDER=map_010_density_medium_dynamic_speed_2mps \
bash launch_evaluation.bash 1 vision fixed_env offset=0
```

该命令在一张已生成的森林场景中运行一次单帧模型，结果写入 `evaluation.yaml`。将 `offset` 改为 `1` 或 `2` 可切换另外两种输入模型。

### 4. 运行消融实验

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
7个条件 × 10张地图 × 5个phase seed = 350轮
```

三个模型合计 **1050轮**。汇总指标包括：

- 成功率：到达终点且全程零碰撞的rollout比例。
- 碰撞率：至少发生一次碰撞的rollout比例。
- 飞行时间：只统计成功rollout，并报告均值和中位数。
- 成功率的95% bootstrap CI。
- 相同case下模型间的paired difference。

```bash
python3 envtest/benchmark/run_benchmark.py \
  --config envtest/benchmark/configs/forest_benchmark_v1.yaml \
  --cases envtest/benchmark/manifests/ablation_validation_cases.csv \
  --policy single \
  --policy adjacent \
  --policy skip_one \
  --output results/forest_ablation_v1 \
  --resume

python3 envtest/benchmark/summarize_results.py \
  --results results/forest_ablation_v1/results.csv \
  --output results/forest_ablation_v1
```

汇总后生成三张图：

```text
ablation_dynamic_speed.png
ablation_forest_density.png
ablation_flight_speed.png
```

每张图的横轴是对应因素的三个取值，三条曲线对应 `single`、`adjacent` 和 `skip_one`。图中从上到下依次为带95% bootstrap CI的成功率、碰撞率和成功rollout的中位飞行时间；没有成功rollout时，飞行时间点留空。

主要参数：

- `--config`：场景、模型和评价标准配置。
- `--cases`：本次实验使用的固定 case manifest。
- `--policy`：需要运行的模型，可重复指定。
- `--output`：结果目录。
- `--resume`：跳过结果目录中已经完成的 `(policy_id, case_id)`。
- `--scenario <name>`：可选，只运行指定条件，例如 `dynamic_speed_1mps`。

汇总脚本不自动排序或选择模型。结合 `summary.csv`、`paired_model_differences.csv` 和三张图，人工决定用于主对比实验的模型。

### 5. 运行主对比实验

该实验在未参与消融的 test maps 上，将人工选择的模型与官方 ViTFly 进行配对比较，并测试飞行速度、森林密度和动态障碍速度条件。先将 `SELECTED_POLICY` 设置为人工选择的 `single`、`adjacent` 或 `skip_one`。

```bash
SELECTED_POLICY=adjacent

python3 envtest/benchmark/run_benchmark.py \
  --config envtest/benchmark/configs/forest_benchmark_v1.yaml \
  --cases envtest/benchmark/manifests/comparison_test_cases.csv \
  --policy "$SELECTED_POLICY" \
  --policy original_vitfly \
  --output results/main_comparison_v1 \
  --resume

python3 envtest/benchmark/summarize_results.py \
  --results results/main_comparison_v1/results.csv \
  --output results/main_comparison_v1
```

结果包括每个 rollout 的记录，以及按模型和场景汇总的成功率、碰撞率、成功轨迹飞行时间、95% bootstrap CI 和 paired model difference。修改 checkpoint、配置或 manifest 后应使用新的结果目录。

## Gather dataset

数据采集使用可访问完整状态的动态 A* expert，训练输入仍为深度图及对应监督标签。

### 1. 生成动态采集环境

该步骤要求原始 `spheres_medium/environment_0..100` 已放置完成：

```bash
python3 envtest/ros/generate_dynamic_astar_env.py \
  --source-level spheres_medium \
  --env-ids 0-100 \
  --num-dynamic 8 \
  --difficulty medium \
  --seed 10
```

生成器不会覆盖已有目录。如需重新生成，确认旧数据不再需要后增加 `--overwrite`。

### 2. 采集 expert 轨迹

```bash
bash launch_evaluation.bash 10 state
```

`10` 是本批次轨迹数，可按需要修改。采集结果写入 `envtest/ros/train_set/`。每批结束后脚本会自动删除 rejected 轨迹，并更新 `collection_summary.json`。

### 3. 校验数据

```bash
PYTHONPATH=envtest/ros python3 envtest/ros/validate_dataset.py \
  envtest/ros/train_set \
  --require-env-fields \
  --require-multiple-envs
```

### 4. 转移 accepted 轨迹

```bash
mkdir -p training/datasets/dataset
find envtest/ros/train_set \
  -mindepth 1 -maxdepth 1 -type d \
  -exec cp -a {} training/datasets/dataset/ \;
```

只复制轨迹目录，不复制 `collection_summary.json`。

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
