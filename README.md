# SpatialMind

SpatialMind 使用真实图片判断英文空间关系描述的真假。项目以 Qwen3.5-2B 为基础模型，分别在 VSR Random 和 Zero-shot 划分上使用 [ms-swift](https://github.com/modelscope/ms-swift) 做 LoRA 微调，比较微调前后的表现，并提供本地 Gradio Demo。完整配置、混淆矩阵和关系分析见[实验结果报告](reports/experiments.md)。

## 项目流程

1. **准备数据**：按官方划分读取 VSR 元数据与 COCO 图片，校验图片和标签，再转换为 ms-swift 可读的 `messages`＋`images` JSONL。Random 为训练／验证／测试 7,680／1,097／2,195 条；当前 [Zero-shot 数据文件](https://huggingface.co/datasets/cambridgeltl/vsr_zeroshot)为 3,489／340／1,222 条，原始 `dev` 在转换后命名为 `validation`。作者仓库 README 的旧 Zero-shot 数量与当前文件不同，本项目按下载的文件运行。
2. **建立 Base 基线**：用同一套提示词和解码设置评测未经微调的 Qwen3.5-2B。
3. **独立训练 LoRA**：Random 和 Zero-shot 各自使用自己的训练集、验证集与 Adapter，按验证 loss 选择 checkpoint。
4. **配对评测与分析**：在各自完整测试集上比较 Base／LoRA 的 Accuracy、Precision、Recall、F1，按空间关系类别统计并人工复核错误案例。

图片不包含在 VSR 元数据文件内，需按[VSR 数据说明](https://github.com/cambridgeltl/visual-spatial-reasoning/tree/master/data)单独准备。测试集不参与训练和选 checkpoint。Random 与 Zero-shot 的训练规模、关系分布和测试样本不同，跨划分的分数差异不能单独归因于物体概念是否见过。

## 使用 ms-swift 微调

### 1. 准备多模态样本

[`scripts/prepare_dataset.py`](scripts/prepare_dataset.py)把一条“图片＋caption＋真假标签”转换成 ms-swift [自定义多模态数据格式](https://github.com/modelscope/ms-swift/blob/main/docs/source/Customization/Custom-dataset.md)。输出中的每一行大致如下；实际图片路径由脚本写为本机绝对路径：

```json
{"messages":[{"role":"user","content":"<image>\nDoes the caption correctly describe the spatial relation in the image?\nCaption: The cat is on the bed.\nAnswer True or False."},{"role":"assistant","content":"True"}],"images":["/path/to/image.jpg"]}
```

`<image>` 让模型接收图像；assistant 的 `True`／`False` 是监督目标。训练时必须保留真实图片。Random 使用 `--protocol random`（默认），Zero-shot 使用 `--protocol zeroshot`；两者分别生成 `train.jsonl`、`validation.jsonl` 和 `test.jsonl`。

### 2. 调用 `swift sft`

项目的 [`scripts/train_lora.py`](scripts/train_lora.py) 是一个薄封装：检查数据路径与新输出目录，然后调用环境中的 **`swift sft`**，将训练 JSONL 传给 `--dataset`、官方验证 JSONL 传给 `--val_dataset`，并设置 `--split_dataset_ratio 0`，避免再从训练集自动切分。命令来源和参数形式参考 [ms-swift 快速开始](https://github.com/modelscope/ms-swift/blob/main/docs/source/GetStarted/Quick-start.md)。

下面以本机已转换的 Random 数据为例。先运行 `--dry-run` 查看完整的 `swift sft` 命令；正式训练需使用一个**新的空输出目录**。将 `$data` 换成 `D:\SpatialMind\output\vsr_zeroshot`，即可对 Zero-shot 划分独立训练；两组应使用不同的 `$out`。

```powershell
Set-Location D:\SpatialMind
$env:PYTHONNOUSERSITE = '1'
$py = 'C:\Users\Lenovo\.conda\envs\spatialmind\python.exe'
$model = 'D:\models\Qwen3.5-2B-modelscope'
$data = 'D:\SpatialMind\output\vsr_random'
$out = 'D:\SpatialMind\output\lora_random_new_run'
& $py -s .\scripts\train_lora.py --model $model --train-jsonl "$data\train.jsonl" --val-jsonl "$data\validation.jsonl" --output-dir $out --epochs 1 --dry-run
& $py -s .\scripts\train_lora.py --model $model --train-jsonl "$data\train.jsonl" --val-jsonl "$data\validation.jsonl" --output-dir $out --epochs 1
```

`train_lora.py` 向 ms-swift 传入的关键设置：

| ms-swift 参数 | 本项目设置与作用 |
|---|---|
| `--model_type qwen3_5`、`--template qwen3_5` | 使用与 Qwen3.5-2B 对应的模型类型和对话模板。 |
| `--tuner_type lora`、`--lora_rank 8`、`--lora_alpha 32`、`--target_modules all-linear` | 对语言部分的线性层添加 LoRA，训练少量新增参数。 |
| `--freeze_vit true`、`--freeze_aligner true` | 冻结视觉编码器和视觉语言对齐器，降低 8GB 显存负担；图片仍参与前向计算。 |
| `--per_device_train_batch_size 1`、`--gradient_accumulation_steps 4` | 每次送入 1 条，累积 4 次梯度后更新一次。 |
| `--torch_dtype bfloat16`、`--gradient_checkpointing true`、`IMAGE_MAX_TOKEN_NUM=512` | 控制训练精度、激活显存和图像 token 数量。 |
| `--learning_rate 1e-4`、`--num_train_epochs 1`、`--seed 42` | 两组实验使用相同的起始训练设置。 |
| `--add_non_thinking_prefix true`、`--loss_scale ignore_empty_think` | 适配 Qwen3.5 的非思考回答及训练损失。 |
| `--eval_steps 100`、`--save_steps 100`、`--save_total_limit 2` | 定期验证并保存，最多保留两个 checkpoint；取日志中验证 loss 最低的 `best_model_checkpoint`。 |

可以先加 `--smoke` 用 200 条训练和 32 条验证检查流程；它也需要单独的新输出目录。完整训练已在本地 RTX 4060 Laptop GPU（8GB）完成：Random 最佳为 checkpoint-1700，Zero-shot 最佳为 checkpoint-800。显存需求会随模型和图像 token 数变化；若扩大配置，先用小样本测峰值显存，再决定是否租云 GPU。本项目现有配置不需要云服务器。

## 结果

两组均在各自完整测试集上评测；同一划分内的 Base／LoRA 使用相同样本、提示词、图像 token 上限 512、最多生成 8 token 和 temperature 0，均无无效输出。

| 划分 | Base Accuracy / F1 | LoRA Accuracy / F1 | Accuracy 提升 |
|---|---:|---:|---:|
| Random（2,195 条） | 73.17% / 70.30% | 85.60% / 86.34% | +12.44 个百分点 |
| Zero-shot（1,222 条） | 77.00% / 75.20% | 83.80% / 85.29% | +6.79 个百分点 |

Zero-shot 的 LoRA Recall 上升而 Precision 下降：假阳性从 78 增至 143。8 类关系结果、完整指标与训练记录见[实验结果报告](reports/experiments.md)；看图复核见 [Random 报告](reports/error_analysis.md)和 [Zero-shot 报告](reports/zeroshot_error_analysis.md)。两组都只训练了一个随机种子，部分关系样本较少；这些限制在报告中说明。

## 本地前端 Demo

[`demo/app.py`](demo/app.py) 使用 Gradio 展示图片上传、英文空间关系描述输入和 Base／LoRA 切换。以下命令使用已完成的 Random Adapter：

```powershell
$env:PYTHONNOUSERSITE = '1'
& 'C:\Users\Lenovo\.conda\envs\spatialmind\python.exe' -s 'D:\SpatialMind\demo\app.py' --model 'D:\models\Qwen3.5-2B-modelscope' --adapter-path 'D:\SpatialMind\output\lora_qwen35_random_e1\v0-20261030-191530\checkpoint-1700'
```

保持这个终端窗口打开，再在**同一台电脑**访问 [http://127.0.0.1:7860](http://127.0.0.1:7860)。页面支持上传图片、输入英文空间关系描述，以及切换 Base／LoRA。8GB 显卡上切换模式会先释放旧模型，再加载新模型。

## 代码与产物

- `scripts/`：数据转换、ms-swift 训练入口、评测与错误案例导出。
- `src/`：统一推理、指标计算和关系类别映射。
- `demo/`：本地 Gradio 前端。
- `reports/`：可纳入 Git 的实验结果和人工复核报告。
- `output/`：本机 JSONL、逐条预测、日志和 Adapter；由 `.gitignore` 排除，不随仓库发布。

本机环境：Python 3.11、PyTorch 2.6.0+cu126、Transformers 5.2.0、ms-swift 4.7.0.dev0。依赖见 [`requirements.txt`](requirements.txt)。本机 ms-swift 是无 Git 元数据的源码快照，精确复现需要保留该快照；运行时使用 `PYTHONNOUSERSITE=1` 和 `python -s`，避免用户目录中的 CPU 版 PyTorch 遮蔽 CUDA 环境。
