# ColBERT 本地数据与运行说明

在项目根目录 `multi-vector` 中执行命令，使用已经安装依赖的虚拟环境。

## 数据目录

```text
data/
  msmarco/
    collection.tsv                  # 尚缺：MS MARCO 训练文档
    train/
      queries.tsv                   # 已提供：训练问题
      hard_negatives_32.tsv         # 已提供：训练样本 ID
  nanobeir_merged/
    corpus.jsonl                    # 评测文档
    queries.jsonl                   # 评测问题
    qrels/
      test.tsv                      # 评测相关性标注
    original/
      corpus.jsonl                  # 原始评测语料备份，内容与工作副本相同
```

训练样本中的 MS MARCO 数字文档 ID，与 NanoBEIR 评测文档的带前缀 ID 不同，不能用评测语料替代训练用的 `collection.tsv`。

`collection.tsv` 必须是与训练样本对应的 MS MARCO Passage 语料，每行格式为 `文档ID<Tab>段落文本`，无表头；解压后放到 `data/msmarco/collection.tsv`。

## 训练

补齐训练语料后运行：

```powershell
python train.py --config configs/train/colbert_windows.yaml
```

新配置使用基础 BERT 初始化，不需要已有 ColBERT 权重。它按一个训练轮次运行，并使用提供的 NanoBEIR 数据每 1000 个优化步骤评测一次，训练结束也会评测。若此测试集用于选择模型，后续不能将其当作独立、未参与模型选择的最终测试集。

4 GB 显存的起始设置：训练批量 1、每个问题 1 个正样本和 1 个负样本、文档长度 128、FP16、梯度累积 16。梯度累积不等同于扩大批内负样本池。这些参数用于本机起步，并非原论文实验参数，也不保证所有环境下都不会显存不足。

想先检查短训练，可把训练 YAML 的 `max_steps` 改为 `20`、`warmup_steps` 改为 `0`、`logging_steps` 改为 `1`。注意结束时仍会运行一次全量评测。正式训练时恢复 `max_steps: -1`，并按需要设置预热步骤。重复运行默认会从基础 BERT 重新训练；要保留不同实验，应设置不同的 `output_dir`。

当前代码会将训练问题、完整文档语料和训练样本读入内存；全量 MS MARCO 也需要足够的系统内存。开始长时间训练前应检查内存占用。

最终模型保存在 `checkpoints/colbert_windows/model.safetensors`；中间训练状态位于该目录下的 `checkpoint-*` 子目录。配置中的 `logs/colbert_windows` 预留作日志目录；`report_to: none` 不启用 W&B 或 TensorBoard，训练进度显示在终端，指标和状态写入模型输出目录。

## 单独评测

训练成功保存最终权重后运行：

```powershell
python evaluate.py --config configs/evaluate/colbert_windows.yaml --progress
```

评测配置与训练配置使用相同的模型结构和长度设置，结果保存到 `result/colbert_windows.json`。

## 文件检查说明

提供的训练问题与训练样本的查询 ID 已检查匹配。评测语料中含 Unicode 行分隔字符，不能将它们误当作 JSONL 记录边界；文件内容保持原样，并在 `original/` 中保留备份。训练语料尚未提供，因此无法验证训练文档 ID 是否齐全，也尚未运行正式训练。
