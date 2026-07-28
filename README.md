# ViTFly Dynamics：动态森林中的视觉无人机避障

[原项目主页](https://www.anishbhattacharya.com/research/vitfly) ·
[原论文](https://arxiv.org/abs/2405.10391)

本项目在 ViTFly 基础上实现动态森林中的多帧避障训练和统一 Benchmark。学习策略包括单帧、相邻双帧和隔帧双帧，并支持与官方 ViTFly、FastPlanner 和 EGO-Planner 对比。

## 目录

- [安装](#安装)
- [资产](#资产)
- [模型](#模型)
- [Benchmark](#benchmark)
- [数据采集](#数据采集)
- [训练](#训练)
- [引用](#引用)

## 安装

仿真环境以 Ubuntu 20.04、ROS Noetic 和 Python 3.8 为基准。

```bash
source /opt/ros/noetic/setup.bash
sudo apt update
sudo apt install -y python3-catkin-tools

mkdir -p ~/catkin_ws/src
cd ~/catkin_ws
catkin init
catkin config --extend /opt/ros/noetic
catkin config --merge-devel
catkin config --cmake-args -DCMAKE_BUILD_TYPE=Release

cd ~/catkin_ws/src
git clone --branch code-release --single-branch --recurse-submodules \
  git@github.com:astringent-orange/vitfly-dynamics.git
cd vitfly-dynamics
bash setup_ros.bash

cd ~/catkin_ws
catkin build
```

创建 Python 环境。PyTorch 版本需与本机 CUDA 匹配：

```bash
conda create -n pubflight python=3.8 pip -y
conda activate pubflight
python -m pip install --upgrade pip
pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r ~/catkin_ws/src/vitfly-dynamics/requirements.txt
```

每次打开新终端后进入环境：

```bash
conda activate pubflight
source ~/catkin_ws/devel/setup.bash
cd ~/catkin_ws/src/vitfly-dynamics
```

## 资产

所有百度网盘资源的提取码均为 `9527`。

### 场景

下载 [environments.tar.xz](https://pan.baidu.com/s/14q8FcqlNW3rIg7haxARpHw?pwd=9527)，在仓库根目录执行：

```bash
tar -xJf <path/to/environments.tar.xz> \
  -C flightmare/flightpy/configs/vision
```

### Unity 渲染器

下载 [flightrender.tar.xz](https://pan.baidu.com/s/19u9rlteO3mBndggyONYz7A?pwd=9527)：

```bash
tar -xJf <path/to/flightrender.tar.xz> -C flightmare/flightrender
```

### 预训练模型

下载 [pretrained_models.tar.xz](https://pan.baidu.com/s/13EyXOon9YnWpx5HPMuTMmw?pwd=9527)：

```bash
tar -xJf <path/to/pretrained_models.tar.xz> -C models
```

### ViTFly 专家数据模型

下载 [pretrained_models_vitfly.tar.xz](https://pan.baidu.com/s/1LXIxw2Hf_26VER9YgrgEzA?pwd=9527)（提取码：`9527`）：

```bash
tar -xJf <path/to/vitfly_expert_models.tar.xz> -C models
```

### 训练数据集

下载 [dataset.tar.xz](https://pan.baidu.com/s/17TVpN4KkvG-Y7gJA54uheg?pwd=9527)：

```bash
mkdir -p training/datasets
tar -xJf <path/to/dataset.tar.xz> -C training/datasets
```

## 模型

默认模型由 Benchmark 配置文件管理：

| Policy | 输入 | 权重位置 |
|---|---|---|
| `single` | 当前帧 | `models/current_frame/current_frame_vitlstm_000099.pth` |
| `adjacent` | 当前帧和前一帧 | `models/previous_frame/previous_frame_vitlstm_000099.pth` |
| `skip_one` | 当前帧和前两帧 | `models/second_previous_frame/second_previous_frame_vitlstm_000099.pth` |
| `original_vitfly` | 官方 ViTFly | `models/ViTLSTM_model.pth` |
| `vitfly_data_single` | ViTFly 专家数据，当前帧 | `models/current_frame_vitfly/current_frame_vitlstm_000099.pth` |
| `vitfly_data_adjacent` | ViTFly 专家数据，当前帧和前一帧 | `models/previous_frame_vitfly/previous_frame_vitlstm_000099.pth` |
| `vitfly_data_skip_one` | ViTFly 专家数据，当前帧和前两帧 | `models/second_previous_frame_vitfly/second_previous_frame_vitlstm_000099.pth` |

对应配置为 `envtest/benchmark/configs/forest_benchmark_vitfly_data_v1.yaml`，policy 名称为 `vitfly_data_single`、`vitfly_data_adjacent` 和 `vitfly_data_skip_one`。

训练生成的 checkpoint 必须与同一次训练生成的 `run_metadata.json` 放在同一目录。修改文件名或目录后，同步更新相应 YAML 中的 `checkpoint`。

运行一次可视化测试：

```bash
VITFLY_ENV_LEVEL=forest_benchmark_v1 \
VITFLY_ENV_FOLDER=map_010_density_6_dynamic_speed_2mps \
bash launch_evaluation.bash 1 vision fixed_env offset=0
```

`offset=0/1/2` 分别使用单帧、相邻双帧和隔帧双帧模型；`model_path=<checkpoint>` 可显式指定其他权重。

## Benchmark

### 测试条件

森林密度固定为 `6 棵/100 m²`。正式实验包含两组单变量扫描：

| 扫描变量 | 五档取值 | 固定条件 |
|---|---|---|
| 动态障碍物速度 | `1 / 2 / 3 / 4 / 5 m/s` | 无人机期望速度 `5 m/s` |
| 无人机期望速度 | `2 / 4 / 6 / 8 / 10 m/s` | 动态障碍物速度 `2 m/s` |

每个条件使用 10 张地图和 5 个 phase seed：

```text
10个条件 × 10张地图 × 5个phase seed = 500轮/模型
```

生成场景和固定测试清单：

```bash
python3 envtest/benchmark/generate_forest_scenes.py \
  --config envtest/benchmark/configs/forest_benchmark_v1.yaml

python3 envtest/benchmark/build_manifest.py \
  --config envtest/benchmark/configs/forest_benchmark_v1.yaml \
  --output envtest/benchmark/manifests
```

Manifest 是固定的 CSV 测试清单，每行定义一次 rollout 的地图、速度和 phase seed。所有模型使用相同 case，以便配对比较。

### 消融实验

在 validation maps 上分别运行三个模型，每条命令默认完成 500 轮：

```bash
python3 envtest/benchmark/run_benchmark.py --policy single
python3 envtest/benchmark/run_benchmark.py --policy adjacent
python3 envtest/benchmark/run_benchmark.py --policy skip_one
```

结果默认写入 `results/ablation/<policy>_<时间>/`。三个模型完成后汇总：

```bash
python3 envtest/benchmark/summarize_results.py
```

使用 ViTFly 专家数据训练的模型时指定另一份配置：

```bash
python3 envtest/benchmark/run_benchmark.py \
  --config envtest/benchmark/configs/forest_benchmark_vitfly_data_v1.yaml \
  --policy vitfly_data_single

python3 envtest/benchmark/run_benchmark.py \
  --config envtest/benchmark/configs/forest_benchmark_vitfly_data_v1.yaml \
  --policy vitfly_data_adjacent

python3 envtest/benchmark/run_benchmark.py \
  --config envtest/benchmark/configs/forest_benchmark_vitfly_data_v1.yaml \
  --policy vitfly_data_skip_one
```

### 主对比实验

主实验在 test maps 上运行，每个模型同样测试五档障碍物速度和五档飞行速度，共 500 轮。将人工选出的学习模型与三个基线分别运行：

```bash
python3 envtest/benchmark/run_comparison.py --policy adjacent
python3 envtest/benchmark/run_comparison.py --policy vitfly
python3 envtest/benchmark/run_comparison.py --policy fastplanner
python3 envtest/benchmark/run_comparison.py --policy egoplanner
```

`single`、`adjacent` 或 `skip_one` 均可作为人工选定模型，主实验结果中统一记录为 `best_ours`。四个模型完成后执行：

```bash
python3 envtest/benchmark/summarize_comparison.py
```

### 恢复与筛选

中断后使用原输出目录恢复：

```bash
python3 envtest/benchmark/run_benchmark.py \
  --policy single \
  --output results/ablation/single_YYYYMMDD_HHMMSS \
  --resume
```

主对比同样支持 `--output ... --resume`。常用参数：

- `--scenario <name>`：只运行指定条件，可重复使用。
- `--case-id <id>`：只运行指定 case，可重复使用。
- `--limit <n>`：限制本次 case 数量。
- `--no-reuse-simulator`：关闭同场景仿真器复用。
- `--real-time-factor <n>`：覆盖默认 `1.5` 倍仿真速度。

### FastPlanner 与 EGO-Planner

规划器源码以 submodule 固定在 `third_party/`。首次使用时执行：

```bash
git submodule update --init --recursive
bash envtest/fastplanner/build_fastplanner.bash
bash envtest/egoplanner/build_egoplanner.bash
```

构建后检查运行环境：

```bash
python3 envtest/benchmark/preflight.py \
  --cases envtest/benchmark/manifests/comparison_test_cases.csv \
  --policy fastplanner --policy egoplanner
```

输出 `ready: true` 后再运行主对比命令。

## 数据采集

生成动态 A* expert 场景：

```bash
python3 envtest/ros/generate_dynamic_astar_env.py \
  --source-level spheres_medium \
  --env-ids 0-100 \
  --num-dynamic 8 \
  --difficulty medium \
  --seed 10
```

采集指定数量的 A* expert 轨迹：

```bash
bash launch_evaluation.bash 10 state
```

采集结果位于 `envtest/ros/train_set/`。系统会拒绝并删除碰撞、未到达终点、越界、超时、图像与 CSV 数量不匹配、时间戳非递增或包含非法深度帧的轨迹。

采集官方 ViTFly expert 数据：

```bash
bash envtest/ros/collect_vitfly_ablation.bash
```

## 训练

数据集目录结构为：

```text
training/datasets/<dataset_name>/<trajectory>/data.csv
training/datasets/<dataset_name>/<trajectory>/*.png
```

从仓库根目录训练三种输入模型：

```bash
CUDA_VISIBLE_DEVICES=0 python3 training/train.py \
  --dataset <dataset_name> --offset 0

CUDA_VISIBLE_DEVICES=0 python3 training/train.py \
  --dataset <dataset_name> --offset 1

CUDA_VISIBLE_DEVICES=0 python3 training/train.py \
  --dataset <dataset_name> --offset 2
```

训练参数默认读取 `training/config/train_vitlstm.txt`，输出位于 `training/logs/`。查看训练曲线：

```bash
tensorboard --logdir training/logs
```

将最终 checkpoint 和 `run_metadata.json` 放入 [模型](#模型) 中对应目录，并更新 Benchmark 配置中的权重路径。

## 引用

```bibtex
@inproceedings{bhattacharya2025vision,
  title={Vision transformers for end-to-end vision-based quadrotor obstacle avoidance},
  author={Bhattacharya, Anish and Rao, Nishanth and Parikh, Dhruv and Kunapuli, Pratik and Wu, Yuwei and Tao, Yuezhan and Matni, Nikolai and Kumar, Vijay},
  booktitle={2025 IEEE International Conference on Robotics and Automation (ICRA)},
  year={2025},
  organization={IEEE}
}
```

仿真基础来自 [ICRA 2022 DodgeDrone Competition](https://github.com/uzh-rpg/agile_flight)。
