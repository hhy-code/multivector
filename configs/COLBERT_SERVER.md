# 原始参数的 ColBERT 服务器配置

`train/colbert_server.yaml` 和 `evaluate/colbert_server.yaml` 分别对应原项目的两份 `colbert.yaml`。所有非路径参数均保留原值，没有按本地笔记本或 A5000 降低配置，也没有添加短训练步数上限。

## 数据与路径

在服务器项目根目录 `multi-vector` 下，训练所需文件为：

```text
data/msmarco/collection.tsv
data/msmarco/train/queries.tsv
data/msmarco/train/hard_negatives_32.tsv
data/nanobeir_merged/corpus.jsonl
data/nanobeir_merged/queries.jsonl
data/nanobeir_merged/qrels/test.tsv
```

创建这份配置时，本地缺少 `data/msmarco/collection.tsv`。服务器上需要补齐与训练样本 ID 对应的语料。数据放在其他位置时，修改 YAML 中对应的路径即可。项目读取的是解压后的数据文件。

原配置的 `nway: 1` 表示每个问题取正样本文档，不显式抽取文件中的负样本文档；训练利用同批次其他文档作为负样本，启用分布式训练时还会收集跨卡文档。这是原项目的设置。

## 启动训练

在服务器准备好 Python、GPU 版 PyTorch 及项目依赖，并激活对应环境；Windows 的 `.venv` 不能直接移到 Linux 使用。原配置启用了 W&B，在服务器终端完成登录：

```bash
wandb login
```

从项目根目录启动单卡训练：

```bash
python train.py --config configs/train/colbert_server.yaml
```

如果使用 Linux 多卡服务器，例如 2 张 GPU：

```bash
torchrun --standalone --nproc_per_node=2 train.py --config configs/train/colbert_server.yaml
```

将 `2` 替换为实际使用的 GPU 数量。配置里的批量 128 是每张卡的训练批量；显卡数量会影响全局批量与每轮的优化步骤数量。

完整训练为 4 轮，每 500 个优化步骤评测并保存一次，每 50 步记录日志。训练过程中模型位于 `checkpoints/colbert_0927/states/checkpoint-*`，训练结束的模型位于 `checkpoints/colbert_0927/states/model.safetensors`。训练结束时现有代码会恢复已保存检查点中评测最好的模型后保存最终权重。

原配置没有限制保留检查点的数量，服务器磁盘会保存多个检查点。断点续训时可在训练配置中启用 `resume_from_checkpoint: true`，从输出目录的最新检查点恢复；也可指定一个实际存在的检查点目录。

## 单独评测

原项目训练期间使用 NanoBEIR 合并集，独立评测配置则使用 BEIR 的 `webis-touche2020`；新配置保留这种安排。

```bash
python evaluate.py --config configs/evaluate/colbert_server.yaml --progress
```

该命令默认加载原配置所选的第 8000 步检查点，因此必须先有 `checkpoints/colbert_0927/states/checkpoint-8000/model.safetensors`。多卡训练总步数可能不同，不一定会产生该目录。要评测最终保存的模型，手动把评测 YAML 中的 `ckpt_path` 改成 `./checkpoints/colbert_0927/states/model.safetensors`。

`data/beir/webis-touche2020` 不存在或为空时，现有评测代码会尝试联网下载数据集。已有的 `data/nanobeir_merged` 不等同于此数据集。以上是原项目配置的路径适配，并不意味着严格复现原始 ColBERT 论文实验。
