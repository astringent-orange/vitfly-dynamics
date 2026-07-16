# ViTFly Dynamics：动态森林中的视觉无人机避障

[原项目主页](https://www.anishbhattacharya.com/research/vitfly) &nbsp;
[原论文](https://arxiv.org/abs/2405.10391)

本项目基于 ICRA 2025 ViTFly，保留官方 ViT+LSTM 作为对比模型，并扩展为面向动态森林的行为克隆与统一 Benchmark。当前学习策略包括单帧、相邻双帧和隔一帧双帧三种输入；测试支持飞行速度、森林密度和动态障碍速度的可复现实验条件。

仓库只保存代码和小型配置。Flightmare/Unity、森林场景、训练数据和模型权重属于大型资产，需要单独放置。

## Installation

完整仿真环境以 **Ubuntu 20.04 + ROS Noetic** 为基准。只训练模型时可以跳过 ROS、catkin、Flightmare 和 Unity，直接阅读 [Train](#train)。训练脚本需要 NVIDIA GPU；仿真环境建议使用与 ROS Noetic 兼容的 Python 3.8。

#### 安装 ROS 与 catkin tools

先按 ROS 官方方式安装 ROS Noetic，然后执行：

```bash
source /opt/ros/noetic/setup.bash
sudo apt update
sudo apt install -y python3-catkin-tools
```

#### 创建 catkin 工作空间

如果还没有工作空间，按以下方式创建：

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
```

#### 克隆代码

在 `catkin_ws/src` 下将仓库目录命名为 `vitfly`：

```bash
cd ~/catkin_ws/src
git clone --branch code-release --single-branch \
  git@github.com:astringent-orange/vitfly-dynamics.git vitfly
cd vitfly
```

如果使用已有副本：

```bash
git switch code-release
git pull --ff-only origin code-release
```

#### 放置大型资产（待补充）

> TODO：此处只规定资产的目标位置。下载地址、版本、文件名和校验和由维护者在发布资产后补充。

| 大型资产 | 必须放置到 | 用途 |
|---|---|---|
| Flightmare Unity renderer | `flightmare/flightrender/` | 启动 Unity 图形仿真 |
| 原始 `trees` 场景 | `flightmare/flightpy/configs/vision/trees/` | 生成森林 Benchmark 场景 |
| 原始 `spheres_medium` 场景 | `flightmare/flightpy/configs/vision/spheres_medium/` | 生成动态 A* 采集场景 |
| 官方 ViT+LSTM 权重 | `models/ViTLSTM_model.pth` | 原始 ViTFly 对比实验 |
| 三个训练权重及各自的 `run_metadata.json` | `models/current_frame/`、`models/previous_frame/`、`models/second_previous_frame/` | 三种输入消融与主实验 |
| 筛选后的训练数据集 | `training/datasets/dataset/` | 模型训练 |

资产解压命令占位：

```bash
# TODO：将下列占位符替换为实际资产包路径
tar -xf <environment-archive.tar> -C flightmare/flightpy/configs/vision
tar -xf <renderer-archive.tar> -C flightmare/flightrender

# 数据解压后，dataset/ 的直接子目录应当是各条 trajectory 目录
unzip <dataset-archive.zip> -d training/datasets/dataset
```

最终目录至少应包含：

```text
models/
├── ViTLSTM_model.pth
├── current_frame/
│   ├── current_frame_vitlstm_000099.pth
│   └── run_metadata.json
├── previous_frame/
│   ├── previous_frame_vitlstm_000099.pth
│   └── run_metadata.json
└── second_previous_frame/
    ├── second_previous_frame_vitlstm_000099.pth
    └── run_metadata.json
```

`run_metadata.json` 用于校验 checkpoint 的模型结构和帧偏移。官方旧权重没有该文件，由 `vitfly_legacy` adapter 显式兼容；新训练的三个模型必须保留该文件。

#### 安装依赖并构建

从仓库根目录运行安装脚本，再回到 catkin 工作空间构建：

```bash
cd ~/catkin_ws/src/vitfly
bash setup_ros.bash

cd ~/catkin_ws
catkin build
source devel/setup.bash
cd src/vitfly
```

为训练和 Python 节点创建环境。完整 ROS 仿真建议使用 Python 3.8；只在服务器训练时可使用与 PyTorch/CUDA 匹配的 Python 3.10：

```bash
conda create -n vitfly python=3.8 pip -y
conda activate vitfly
python -m pip install --upgrade pip

# 先按服务器 CUDA 版本安装 PyTorch；下面仅为 CUDA 12.1 示例
pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

每次打开终端后执行：

```bash
conda activate vitfly
source /opt/ros/noetic/setup.bash
source ~/catkin_ws/devel/setup.bash
export FLIGHTMARE_PATH=~/catkin_ws/src/vitfly/flightmare
cd ~/catkin_ws/src/vitfly
```

可将后三条环境命令加入 `~/.bashrc`。确认环境：

```bash
python3 -c "import torch, cv2, yaml; print(torch.__version__, torch.cuda.is_available())"
rospack find envsim
test -x flightmare/flightrender/vitfly-unity.x86_64
```

## Test (simulation)

#### 运行代码级测试

这些测试不启动 Unity，用于检查模型输入、数据加载、场景生成、manifest、评价器和汇总逻辑：

```bash
python3 -m unittest discover -s training -p 'test_*.py' -v
python3 -m unittest discover -s envtest/benchmark/tests -p 'test_*.py' -v
PYTHONPATH=envtest/ros python3 -m unittest discover -s envtest/ros -p 'test_*.py' -v
```

#### 生成 Benchmark 场景和固定场次

该步骤要求 `trees/environment_0..19` 已放置完成。生成器不会修改原始 `trees` 场景：

```bash
python3 envtest/benchmark/generate_forest_scenes.py \
  --config envtest/benchmark/configs/forest_benchmark_v1.yaml

python3 envtest/benchmark/build_manifest.py \
  --config envtest/benchmark/configs/forest_benchmark_v1.yaml \
  --output envtest/benchmark/manifests
```

生成结果位于：

```text
flightmare/flightpy/configs/vision/forest_benchmark_v1/
envtest/benchmark/manifests/ablation_validation_cases.csv
envtest/benchmark/manifests/comparison_test_cases.csv
envtest/benchmark/manifests/scene_manifest.csv
```

#### 手动运行一次视觉策略

与原版代码相同，`launch_evaluation.bash` 会启动 Flightmare、评价节点和视觉控制器。当前版本不再隐式选择 checkpoint，必须同时给出输入 offset 和权重路径：

```bash
VITFLY_ENV_LEVEL=forest_benchmark_v1 \
VITFLY_ENV_FOLDER=map_010_density_medium_dynamic_collection \
VITFLY_DYNAMIC_PHASE_SEED=9000 \
VITFLY_DES_VEL=5 \
bash launch_evaluation.bash 1 vision fixed_env \
  offset=0 \
  model_path=models/current_frame/current_frame_vitlstm_000099.pth
```

其中第一个参数是 rollout 数量，`vision` 表示运行视觉模型，`fixed_env` 表示所有 rollout 使用指定场景。运行结果默认写入根目录的 `evaluation.yaml`；`/debug_img1` topic 可用于查看带预测速度箭头的深度图。

手动运行官方旧权重时需要显式允许 legacy checkpoint：

```bash
VITFLY_ALLOW_LEGACY_CHECKPOINT=1 \
VITFLY_ENV_LEVEL=forest_benchmark_v1 \
VITFLY_ENV_FOLDER=map_010_density_medium_dynamic_collection \
bash launch_evaluation.bash 1 vision fixed_env \
  offset=0 \
  model_path=models/ViTLSTM_model.pth
```

正式实验应使用下述 Benchmark runner，而不是手动反复修改环境变量。

#### 检查命令和 checkpoint 配置

`--dry-run` 检查场景、policy 和启动命令，但不启动 ROS。必须使用独立的临时结果目录；不要把 dry-run 结果和正式结果混在一起：

```bash
python3 envtest/benchmark/run_benchmark.py \
  --config envtest/benchmark/configs/forest_benchmark_v1.yaml \
  --cases envtest/benchmark/manifests/comparison_test_cases.csv \
  --policy ours_single \
  --scenario baseline \
  --limit 3 \
  --output results/dry_run \
  --dry-run

column -s, -t results/dry_run/results.csv
```

#### 运行一个真实加载测试

下面的命令会实际启动 Flightmare、加载权重并完成一个 rollout：

```bash
python3 envtest/benchmark/run_benchmark.py \
  --config envtest/benchmark/configs/forest_benchmark_v1.yaml \
  --cases envtest/benchmark/manifests/comparison_test_cases.csv \
  --policy ours_single \
  --scenario baseline \
  --limit 1 \
  --output results/checkpoint_smoke

column -s, -t results/checkpoint_smoke/results.csv
```

仅测试官方 ViT+LSTM 时把 policy 改为：

```bash
--policy original_vitfly
```

三个新增 checkpoint 对应关系在 `envtest/benchmark/configs/forest_benchmark_v1.yaml` 中定义：

```text
ours_single    -> models/current_frame/                 -> frame_offset 0
ours_adjacent  -> models/previous_frame/                -> frame_offset 1
ours_skip_one  -> models/second_previous_frame/         -> frame_offset 2
original_vitfly -> models/ViTLSTM_model.pth             -> legacy offset 0
```

#### 运行三场 smoke test

真实 smoke test 不要添加 `--dry-run`：

```bash
python3 envtest/benchmark/run_benchmark.py \
  --config envtest/benchmark/configs/forest_benchmark_v1.yaml \
  --cases envtest/benchmark/manifests/comparison_test_cases.csv \
  --policy ours_single \
  --scenario baseline \
  --limit 3 \
  --output results/smoke_test \
  --resume
```

`--resume` 以 `(policy_id, case_id)` 为键跳过已完成项。修改 checkpoint、配置或 manifest 后应换一个新的输出目录，不能继续复用旧结果。

#### 运行消融与主实验

三种输入模型的消融实验：

```bash
python3 envtest/benchmark/run_benchmark.py \
  --config envtest/benchmark/configs/forest_benchmark_v1.yaml \
  --cases envtest/benchmark/manifests/ablation_validation_cases.csv \
  --policy ours_single \
  --policy ours_adjacent \
  --policy ours_skip_one \
  --output results/forest_ablation_v1 \
  --resume

python3 envtest/benchmark/summarize_results.py \
  --results results/forest_ablation_v1/results.csv \
  --select-best \
  --output results/forest_ablation_v1
```

测试消融选出的模型和官方 ViTFly：

```bash
python3 envtest/benchmark/run_benchmark.py \
  --config envtest/benchmark/configs/forest_benchmark_v1.yaml \
  --cases envtest/benchmark/manifests/comparison_test_cases.csv \
  --selected-policy results/forest_ablation_v1/selected_policy.yaml \
  --policy original_vitfly \
  --output results/main_comparison_v1 \
  --resume
```

可使用 `--scenario` 只运行一个条件：

```text
baseline
flight_speed_3
flight_speed_7
forest_density_low
forest_density_high
dynamic_off
dynamic_high
```

每个 rollout 写入 `results.csv` 和独立的 `rollout_logs/`。汇总结果包含成功率、碰撞率、成功轨迹飞行时间、95% bootstrap CI 及 paired model difference。成功定义为到达目标且零碰撞；失败轨迹的超时时间不会混入成功飞行时间均值。

## Gather your own dataset in simulation

数据采集使用可访问完整状态的动态 A* expert，模型训练仍只使用深度图和对应监督标签。

#### 生成动态采集环境

先用少量地图检查资源和几何约束：

```bash
python3 envtest/ros/generate_dynamic_astar_env.py \
  --source-level spheres_medium \
  --env-ids 0-2 \
  --num-dynamic 8 \
  --difficulty medium \
  --seed 10
```

确认成功后生成其余地图，最终得到 `environment_0..100`：

```bash
python3 envtest/ros/generate_dynamic_astar_env.py \
  --source-level spheres_medium \
  --env-ids 3-100 \
  --num-dynamic 8 \
  --difficulty medium \
  --seed 10
```

如果目录已存在，先只校验；确实要替换时才使用 `--overwrite`：

```bash
python3 envtest/ros/generate_dynamic_astar_env.py \
  --env-ids 0-100 \
  --difficulty medium \
  --verify-only
```

#### 运行 expert 采集

先采集一条确认流程：

```bash
bash launch_evaluation.bash 1 state
```

再运行批量采集，例如：

```bash
bash launch_evaluation.bash 10 state
```

默认会依次切换 `dynamic_astar_medium/environment_0..100`，并为每次 rollout 设置可复现的 dynamic phase seed。以下模式可按需使用：

```bash
# 固定在当前指定环境采集
VITFLY_ENV_FOLDER=environment_0 bash launch_evaluation.bash 10 state fixed_env

# 打开 RViz 观察一场
bash launch_evaluation.bash 1 state rviz
```

原始数据写入：

```text
envtest/ros/train_set/<trajectory_id>/data.csv
envtest/ros/train_set/<trajectory_id>/<timestamp>.png
```

**重要：** `launch_evaluation.bash ... state` 在一批采集结束后会自动运行 `curate_dataset.py --apply`，删除本批次中 rejected 的轨迹，并更新 `envtest/ros/train_set/collection_summary.json`。正式采集前应先备份任何需要人工复查的原始轨迹。

#### 校验并转移数据

校验保留下来的数据：

```bash
PYTHONPATH=envtest/ros python3 envtest/ros/validate_dataset.py \
  envtest/ros/train_set \
  --require-env-fields \
  --require-multiple-envs
```

通过后，只复制 trajectory 目录，不复制 `collection_summary.json`：

```bash
mkdir -p training/datasets/dataset
find envtest/ros/train_set \
  -mindepth 1 -maxdepth 1 -type d \
  -exec cp -a {} training/datasets/dataset/ \;
```

训练前建议再次对目标数据目录执行同一校验命令。

## Train

#### 准备数据集

训练数据目录结构必须为：

```text
training/datasets/dataset/
├── <trajectory_1>/
│   ├── data.csv
│   └── <timestamp>.png
├── <trajectory_2>/
│   ├── data.csv
│   └── <timestamp>.png
└── ...
```

每个直接子目录代表一条完整轨迹。训练/验证集按轨迹划分，避免同一轨迹的数据泄漏到两个集合。不要把 rejected 轨迹、嵌套的额外数据集目录或非轨迹目录放入 `dataset/`。

#### 训练三种模型

从仓库根目录运行。`training/config/train_vitlstm.txt` 默认设置数据集、学习率、epoch 数和随机种子；`--offset` 决定输入帧：

```bash
# 单帧：[D_t]
CUDA_VISIBLE_DEVICES=0 python3 training/train.py --offset 0

# 相邻双帧：[D_{t-1}, D_t]
CUDA_VISIBLE_DEVICES=0 python3 training/train.py --offset 1

# 隔一帧双帧：[D_{t-2}, D_t]
CUDA_VISIBLE_DEVICES=0 python3 training/train.py --offset 2
```

也可以显式指定配置文件：

```bash
CUDA_VISIBLE_DEVICES=0 python3 training/train.py \
  --config training/config/train_vitlstm.txt \
  --offset 1
```

当前训练实现要求 CUDA GPU。开始长训练前可用少量轨迹和 epoch 做加载检查：

```bash
CUDA_VISIBLE_DEVICES=0 python3 training/train.py \
  --offset 0 \
  --short 4 \
  --N_eps 1
```

#### 训练输出与恢复

每次训练在 `training/logs/` 下创建独立目录：

```text
training/logs/<run_name>/
├── args.txt
├── config.txt
├── log.txt
├── run_metadata.json
├── train_val_dirs.npy
├── events.out.tfevents.*
└── <model_name>_<epoch>.pth
```

查看曲线：

```bash
tensorboard --logdir training/logs
```

从 checkpoint 恢复时，offset 必须与原模型一致：

```bash
CUDA_VISIBLE_DEVICES=0 python3 training/train.py \
  --offset 1 \
  --load_checkpoint \
  --checkpoint_path training/logs/<run_name>/previous_frame_vitlstm_000099.pth
```

训练结束后，将最终 checkpoint 和同一次运行生成的 `run_metadata.json` 一起复制到对应目录，并更新 `forest_benchmark_v1.yaml` 中的 checkpoint 路径。不要混用不同训练运行的权重与 metadata。

## Development and release

- `main` 用于完整仿真、采集和实验开发；`code-release` 用于发布可训练代码。
- `training/datasets/`、`training/logs/`、三个新增 checkpoint 目录、生成场景和 `results/` 均不进入代码发布。
- FastPlanner 和 EGO-Planner 当前只保留 adapter 名称与统一接口，尚未集成。
- `README-origin.md` 保存原版说明，不作修改。

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

## Debugging tips

#### `catkin build` 报 Eigen 缓存目录不一致

清理 Flightlib 内的 Eigen CMake 缓存，再重新构建：

```bash
cd ~/catkin_ws/src/vitfly/flightmare/flightlib/externals/eigen
rm -rf CMakeCache.txt CMakeFiles
cd ~/catkin_ws
catkin clean
catkin build
```

#### `[Pilot] Not in hover, won't switch to velocity reference!`

如果随后仍出现起飞、hover 和 `start navigation` 日志，该警告通常可以忽略；若一直未进入 hover，检查 ROS topics、Unity renderer 和场景文件是否完整。

#### controller 提前退出

优先检查 checkpoint 路径、`run_metadata.json`、offset 是否匹配，以及当前 Python 环境能否同时导入 PyTorch 和 ROS Python 包。Benchmark 会把这类失败记录为 `controller_error`，不能将其当作普通碰撞结果。
