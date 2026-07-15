# 开发与发布规则

## 分支职责

- 所有代码和文档开发都在 `main`（或从 `main` 创建的 `feat/*` 分支）完成、验证并提交。
- `code-release` 是不含训练数据、模型 checkpoint、Flightmare/Unity 大型资产和运行产物的发布分支。它与 `main` 的历史独立；将已验证的源码提交同步过去时使用明确的 cherry-pick/补丁，不直接合并两个分支。
- 服务器训练只克隆 `code-release`，数据集通过共享存储或 `rsync` 单独传输。

## GitHub 发布权限

- 在执行任何会上传到 GitHub 的命令（尤其是 `git push`）前，必须先说明将发布的分支和提交，并获得用户在当前对话中的明确同意。
- 未获同意时，可以在本地 `main` 提交、在本地更新 `code-release`，但不得推送远程。
- 永远不要提交或推送训练数据、`train_set`、日志、checkpoint、评估输出或仿真大资产。

## 数据集规则

- 训练数据目录只放已接受轨迹；训练加载器直接扫描轨迹目录，不依赖数据集 manifest。
- 每条轨迹的 `data.csv` 是必要监督标签，必须保留。
- 每批采集使用 `envtest/ros/curate_dataset.py --apply` 删除 rejected 轨迹，并更新唯一的 `collection_summary.json`。
