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

#### 3.4 训练数据集（非必要）

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
VITFLY_ENV_FOLDER=map_010_density_6_dynamic_speed_2mps \
bash launch_evaluation.bash 1 vision fixed_env offset=0
```

该命令在一张已生成的森林场景中运行一次单帧模型，结果写入 `evaluation.yaml`。将 `offset` 改为 `1` 或 `2` 可切换另外两种输入模型。

### Ablation experiment

该实验在 validation maps 上使用完全相同的 cases 比较单帧、相邻双帧和隔一帧双帧模型。采用单变量扫描：每次只改变一个因素，另外两个因素保持基准值。

| 扫描因素 | 取值 | 固定条件 |
|---|---|---|
| 动态障碍速度 | 1 / 2 / 3 / 4 / 5m/s | 无人机5m/s、森林密度6棵/100平方米 |
| 无人机速度 | 2 / 4 / 6 / 8 / 10m/s | 森林密度6棵/100平方米、动态障碍2m/s |

动态速度扫描固定无人机5m/s，飞行速度扫描固定动态障碍2m/s，因此实际使用10个条件：

每个生成场景包含6个纵向交互站点，每个站点分别放置低、中、高三个动态障碍，共18个障碍物。
三个高度层覆盖约0.8–9.0m，并使用垂直正弦运动；不同速度profile共享完全相同的空间轨迹，只缩放轨迹时间。

```text
dynamic_speed_1mps
dynamic_speed_2mps
dynamic_speed_3mps
dynamic_speed_4mps
dynamic_speed_5mps
flight_speed_2
flight_speed_4
flight_speed_6
flight_speed_8
flight_speed_10
```

每个模型运行：

```text
10个条件 × 10张地图 × 5个phase seed = 500轮
```

完整运行时每条模型命令会完成500轮。当前已有的8个条件400轮结果保持不变，只需补跑两个高速条件各50轮，共100轮/模型。配置文件中的Benchmark仿真速度为 `1.5×`，即只缩短墙钟时间，不改变仿真中的无人机或障碍物物理速度；可用 `--real-time-factor 1.0` 临时恢复原速。默认按 `scene_id` 复用ROS/Flightmare仿真器：相同地图、森林密度和动态障碍速度的case共用一个仿真会话，每个case仍会重新启动evaluator/controller并重置无人机和动态障碍phase。仿真器启动、必需topic或结果文件发生基础设施故障时，程序默认完整清理并自动重试一次；第二次仍失败则保存诊断并停止整批，避免将环境故障计入模型性能。

同一个 `scene_id` 的多个phase或设定飞行速度会共用一个仿真器；不同地图或动态障碍速度会启动新的仿真会话。复用模式下Runner负责整组仿真器的启动、重启和清理，不能与已经手动启动的ROS/Flightmare实例混用。需要进行隔离模式对照或排查状态残留时，使用 `--no-reuse-simulator` 强制每个case完整重启。

三个模型完整合计 **1500轮**；本次新增部分为300轮。汇总指标包括：

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

只补跑新增高速条件时，使用两个场景筛选（每个模型100轮）：

```bash
python3 envtest/benchmark/run_benchmark.py \
  --policy single \
  --scenario dynamic_speed_5mps \
  --scenario flight_speed_10 \
  --output results/ablation/single_high_speed_YYYYMMDD_HHMMSS \
  --resume
```

将 `single` 替换为 `adjacent` 或 `skip_one`，分别补跑另外两个模型。

需要临时关闭默认的同场景复用时：

```bash
python3 envtest/benchmark/run_benchmark.py \
  --policy single \
  --no-reuse-simulator
```

`--config` 默认使用 `envtest/benchmark/configs/forest_benchmark_v1.yaml`，`--cases` 默认使用 `envtest/benchmark/manifests/ablation_validation_cases.csv`，需要测试其他配置或manifest时仍可显式覆盖。消融实验一次只允许一个 `--policy`。未指定 `--output` 时，结果自动写入带当前时间的目录，例如：

```text
results/ablation/single_20260717_151230/
results/ablation/adjacent_20260717_173510/
results/ablation/skip_one_20260717_195845/
```

三个模型的原有400轮和新增100轮都完成后直接运行：

```bash
python3 envtest/benchmark/summarize_results.py
```

汇总脚本会自动合并每个模型最新的互补结果分片（原400轮和新增100轮）。汇总表格和图像固定写入 `results/ablation/table/`，再次运行会覆盖上一次的汇总结果。需要汇总指定批次时，仍可显式传入多个 `--results` 和一个 `--output`。

汇总后生成两张图：

```text
ablation_dynamic_speed.png
ablation_flight_speed.png
ablation_dynamic_speed_high_speed.png
ablation_flight_speed_high_speed.png
```

前两张图显示完整五档取值；后两张图显示高速区间，动态速度为2/3/4/5m/s、飞行速度为4/6/8/10m/s。三条曲线对应 `single`、`adjacent` 和 `skip_one`。图中从上到下依次为成功率、碰撞率和成功rollout的平均飞行时间；没有成功rollout时，飞行时间点留空。

主要参数：

- `--config`：场景、模型和评价标准配置，默认使用森林Benchmark配置。
- `--cases`：本次实验使用的固定 case manifest，默认使用消融validation manifest。
- `--policy`：本次运行的模型；消融实验只能指定一个。
- `--output`：可选。消融实验默认使用 `results/ablation/<policy>_YYYYMMDD_HHMMSS/`；主对比实验仍需显式指定。
- `--resume`：跳过结果目录中已经完成的 `(policy_id, case_id)`；恢复中断实验时需同时传入原来的 `--output`。
- `--simulator-retries`：基础设施故障后的自动重试次数，默认为 `1`。
- `--real-time-factor`：覆盖配置中的仿真墙钟速度倍率，默认为 `1.5`。
- `--reuse-simulator`：按 `scene_id` 复用仿真器会话；当前配置默认开启。
- `--no-reuse-simulator`：临时关闭复用，强制每个case独立启动仿真器。
- `--scenario <name>`：可选，只运行指定条件，例如 `dynamic_speed_1mps`。
- `--case-id <id>`：可选，只运行manifest中的指定case；可重复传入，用于精确复测并替换异常结果。
- `--rerun-map <id>`：仅与 `--resume` 配合使用，将指定地图已有结果先备份到 `recovery/` 后强制补跑；可重复传入。
- `--rerun-case <id>`：仅与 `--resume` 配合使用，强制补跑指定case；可重复传入。

Benchmark 的每个 case 会在启动日志中记录阶段标记，并把墙钟耗时写入 `results.csv`：
`case_wall_seconds`（整轮）、`simulator_ready_seconds`（仿真器就绪）、
`pilot_prepare_seconds`（飞控初始化至悬停）、`controller_startup_seconds`（控制器启动至导航开始）、
`rollout_wall_seconds`（导航开始至评价完成）和 `cleanup_seconds`（结束清理）。共享仿真器会话的首次 case
另记录 `simulator_session_startup_seconds`。缺失阶段标记留空，不用零值伪造耗时；每次尝试的原始日志位于
`rollout_logs/<policy>__<case_id>__attempt_<n>.log`。

结果中还记录动态交互诊断：无人机高度范围、动态障碍最小距离、动态交互次数、交互持续时间和动态碰撞标记。
汇总表会额外给出动态交互率，帮助确认成功率变化确实来自动态障碍交互，而不是从高度边界绕行。

Benchmark 初始化中的 `off`、`reset_sim` 和 `enable` 不再依赖固定等待，而是等待飞控遥测/状态确认：
分别确认桥接已关闭、状态已复位、桥接已启用。确认超时按仿真器基础设施错误处理，并遵循
`--simulator-retries` 重试策略；复用仿真器时还会校验 `reset_benchmark` 服务返回的 `success` 字段。
普通手动测试仍保留原有的固定等待和 `rostopic pub --once` 行为。Benchmark 默认关闭控制器逐帧推理计时日志，
不会影响模型推理或最终统计；非 Benchmark 可设置 `VITFLY_INFERENCE_TIMING_LOGS=true` 保留该日志。
Benchmark 控制器会保留含少量零像素的有效深度帧，仅拒绝全零、非法尺寸或非有限帧。导航开始后若5秒内没有速度指令，
没有可用深度帧的情况记为可重试的 `input_pipeline_error`；有可用帧但没有指令的情况记为 `controller_error`。
每次尝试的深度帧计数、命令计数和失败详情写入对应的 `__controller.json` 诊断文件。

按 `Ctrl+C` 中断时，当前case不会写入结果；脚本会先清理其进程。之后使用原输出目录继续，例如：

```bash
python3 envtest/benchmark/run_benchmark.py \
  --policy single \
  --output results/ablation/single_YYYYMMDD_HHMMSS \
  --resume
```

`--resume`也会自动重跑并替换已有的 `simulator_error`、`input_pipeline_error`、`runner_timeout` 或 `missing_result` 行，不会产生重复case。
若需要修复某一地图的历史异常结果，可使用 `--resume --rerun-map 4`；runner会先把旧行原子备份到
`recovery/quarantined_results_<时间>.csv`。汇总只接受每个模型完整覆盖manifest中500个唯一case、且不存在基础设施故障或未知退出码的结果；退出码为2的 `controller_error` 是模型失败，允许进入统计。汇总脚本不自动排序或选择模型，需结合 `summary.csv`、`paired_model_differences.csv` 和四张图人工决定用于主对比实验的模型。

### Comparison experiment

该实验在未参与消融的 test maps 上比较人工选出的最优模型、官方ViTFly、FastPlanner和EGO-Planner。采用动态障碍速度和无人机速度两个单因素扫描，森林密度固定为6棵/100平方米：

```text
8个条件 × 10张test地图 × 5个phase seed = 400轮/模型
4个模型 × 400轮 = 1600轮
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

FastPlanner 和 EGO-Planner 分别运行：

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

主对比实验默认也按scene复用仿真器；使用 `--no-reuse-simulator` 可切换为每个case完整重启。两种模式都默认重试一次基础设施故障，并支持 `--simulator-retries`。汇总脚本自动选择四个模型各自最新修改的 `results.csv`，并检查每个模型是否完整包含同一组400个cases且没有基础设施错误；发现中断、不配对或环境故障结果时会要求先恢复实验。验证通过后覆盖写入 `results/comparison/table/`：

```text
summary.csv
summary.json
paired_model_differences.csv
comparison_dynamic_speed.png
comparison_flight_speed.png
```

每张图包含成功率及95% bootstrap CI、碰撞率、成功rollout平均飞行时间三个纵向子图，以及 `best_ours`、`vitfly`、`fastplanner`、`egoplanner` 四条曲线。需要汇总指定批次时，可以重复传入 `--results` 并用 `--output` 指定目录。

#### FastPlanner/EGO-Planner接入接口

当前配置已接入 `fastplanner_ros` 和 `egoplanner_ros`。两个规划器源码放在主仓库外的独立 catkin overlay，避免将大型第三方源码和构建产物提交到本仓库。

固定版本为：

```text
FastPlanner  41be219fe4ecc43bf0e0c2b42a523f8755ccc0bd
EGO-Planner  bfda51284c8c1b476043255a8145ef925a3778a5
NLopt        09b3c2a6da71cabcb98d2c8facc6b83d2321ed71
```

默认路径是主工作空间的兄弟目录，也可以显式设置：

```bash
export VITFLY_FASTPLANNER_WORKSPACE=/absolute/path/to/fastplanner_ws
export VITFLY_EGOPLANNER_WORKSPACE=/absolute/path/to/egoplanner_ws
```

分别构建：

```bash
bash envtest/fastplanner/build_fastplanner.bash
bash envtest/egoplanner/build_egoplanner.bash
```

构建脚本会固定上游 commit、应用目标高度和 EGO 高速兼容补丁，并将本地
`planner_bridge` 链接到对应 overlay。EGO 高速补丁会在实验记录中保留，正式报告中需要披露。

构建后先运行审计：

```bash
python3 envtest/benchmark/preflight.py \
  --cases envtest/benchmark/manifests/comparison_test_cases.csv \
  --policy fastplanner --policy egoplanner
```

只有输出 `ready: true` 才开始长实验。

两个规划器必须满足相同的生命周期接口：

```text
validate(policy_config)
start(case_config)
wait_ready(timeout)
stop()
metadata()
```

当前的 `planner_bridge` 负责订阅本地深度图和 ground-truth odometry，发布目标路径，接收上游 `quadrotor_msgs/PositionCommand`，并转换为本地控制接口：

```text
input:  /kingfisher/dodgeros_pilot/unity/depth
input:  /kingfisher/dodgeros_pilot/groundtruth/odometry
input:  quadrotor_msgs/PositionCommand
output: /kingfisher/dodgeros_pilot/velocity_command
type:   geometry_msgs/TwistStamped
frame:  world
```

接入步骤：

1. 分别构建两个外部 planner overlay。
2. 确认 `planner_bridge` 能收到深度图和 odometry，并发布 ready。
3. 先运行一个 case：

   ```bash
   python3 envtest/benchmark/run_comparison.py \
     --policy fastplanner --limit 1 \
     --output results/comparison/fastplanner_smoke
   ```

4. EGO-Planner 使用相同命令，将 `--policy` 改为 `egoplanner`。
5. 确认 `results.csv` 中没有 `simulator_error`、`runner_timeout` 或 `missing_result`，再运行正式400轮。

Benchmark 下规划器最终都通过 `/kingfisher/dodgeros_pilot/velocity_command` 接入；指南中使用的
`feedthrough_command/dodgeros_msgs::Command` 不适用于当前本地飞控接口。

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
