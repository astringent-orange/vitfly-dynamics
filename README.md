# ViTFly Dynamics

用于动态障碍无人机仿真的深度视觉行为克隆项目。本仓库的 `code-release` 分支只包含训练和实验代码；训练数据、模型 checkpoint 与 Flightmare/Unity 大型环境资源需要单独提供。

## 快速开始：服务器训练

以下流程只训练模型，不需要 ROS、Flightmare 或 Unity。

### 1. 下载代码

服务器配置好 GitHub SSH 密钥后，克隆发布分支：

```bash
git clone --branch code-release --single-branch \
  git@github.com:astringent-orange/vitfly-dynamics.git
cd vitfly-dynamics
```

更新已存在的代码副本：

```bash
git switch code-release
git pull --ff-only origin code-release
```

### 2. 创建 Python 环境

建议通过 Conda 创建独立环境，并使用 Python 3.10（或与服务器 PyTorch/CUDA 组合兼容的版本）：

```bash
conda create -n vitfly-dynamics python=3.10 pip -y
conda activate vitfly-dynamics
python -m pip install --upgrade pip
```

先按 [PyTorch 官方安装选择器](https://pytorch.org/get-started/locally/) 为服务器的 CUDA 驱动安装匹配的 PyTorch wheel，再安装本项目其余训练依赖：

```bash
pip install -r requirements.txt
```

例如，CUDA 12.1 且希望复现当前开发环境的 PyTorch 版本时：

```bash
pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

安装后可检查 GPU 是否可用：

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

### 3. 放置训练数据

将筛选后的数据集放在以下位置（目录名必须与训练配置中的 `dataset` 一致）：

```text
training/datasets/dataset/
├── trajectory_0001/
│   ├── data.csv
│   └── <timestamp>.png
└── trajectory_0002/
    ├── data.csv
    └── <timestamp>.png
```

数据集目录只放筛选后保留的轨迹目录。每条轨迹的 `data.csv` 保留时间戳、姿态、期望速度和专家速度标签，不能删除。

### 4. 开始训练

从仓库根目录执行。单帧与双帧训练使用同一数据集，只有模型及输入帧数不同：

```bash
# 单帧 ViT-LSTM 基线
CUDA_VISIBLE_DEVICES=0 python training/train.py \
  --config training/config/train_vitlstm_1f.txt

# 双帧 ViT-LSTM（[D_{t-0.10s}, D_t]）
CUDA_VISIBLE_DEVICES=0 python training/train.py \
  --config training/config/train_two_frame_vit_lstm.txt
```

训练日志、checkpoint、训练/验证轨迹划分和运行元数据写入 `training/logs/`。服务器长任务建议在 `tmux` 或作业调度器中运行。

一次训练会创建一个按启动时间命名的运行目录，结构示例如下：

```text
training/logs/
└── d07_15_t17_30/
    ├── args.txt                 # 解析后的全部启动参数
    ├── config.txt               # 本次使用的配置文件快照
    ├── log.txt                  # 数据加载、训练和验证输出
    ├── run_metadata.json        # 模型、帧数、数据集和数据划分统计
    ├── train_val_dirs.npy       # 固定的训练/验证轨迹目录划分
    ├── events.out.tfevents.*    # TensorBoard 标量事件
    ├── model_000025.pth         # 按 save_model_freq 保存的 checkpoint
    └── model_000099.pth         # 训练结束时的 checkpoint（epoch 取决于配置）
```

查看训练曲线：

```bash
tensorboard --logdir training/logs
```

## 数据与模型资产

不要将数据集、checkpoint、Flightmare 环境资源或 Unity 二进制文件提交到 Git。它们应通过服务器共享存储、`rsync` 或单独的制品存储传输。发布分支的忽略规则和资产说明见 [ARTIFACTS.md](ARTIFACTS.md)。

## 开发与发布约定

完整的开发与采集环境保留在本地 `main` 分支。代码修改在 `main`（或从它创建的 `feat/*` 分支）完成、测试并提交；随后将已验证的源码改动同步到 `code-release`。在执行任何 `git push` 上传到 GitHub 前，必须先向用户说明将发布的提交并获得明确同意。`code-release` 不包含训练数据和仿真大型资产，服务器仅使用该分支训练。

## 训练实现说明

- `ViTLSTM` 接受单帧深度输入。
- `TwoFrameViTLSTM` 仅把 ViT 的首层输入通道从 1 改为 2；双帧顺序固定为 `[D_{t-0.10s}, D_t]`。
- 序列开始处若没有满足时间间隔的历史帧，加载器跳过该样本，不复制当前帧。
- 训练/验证按轨迹划分，避免同一轨迹泄漏到两个集合。

## 仿真与数据采集

Flightmare/ROS 仿真和数据采集需要额外的环境资源与系统依赖，不属于上述服务器训练最小环境。使用 `bash launch_evaluation.bash <N> state` 完成一批采集后，脚本会自动筛选并删除 rejected 轨迹，同时更新 `envtest/ros/train_set/collection_summary.json`（采集批数、总轨迹数、accepted/rejected 数）。随后将 `train_set/` 中保留的轨迹目录复制到 `training/datasets/dataset/`；不要复制 `collection_summary.json`。
