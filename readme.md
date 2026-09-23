# iFairy

iFairy (also named Fairy±i) is the first 2-bit complex-valued large language model, where all weights are constrained to {±1, ±i}. By introducing complex-valued architectures and a novel quantization scheme, iFairy achieves efficient compression with minimal accuracy loss. Trained from scratch on 100B tokens, it consistently outperforms real-valued 2-bit baselines (BitNet b1.58 and Symmetric INT2) on language modeling and zero-shot benchmarks, and at 1.3B it is the first 2-bit model to surpass a full-precision real-valued model with the same number of parameters.

- Paper: [Complex-domain representation enables accurate and deployable two-bit language models](https://arxiv.org/abs/2508.05571)
- Models: [HuggingFace 700M](https://huggingface.co/PKU-DS-LAB/Fairy-plus-minus-i-700M) · [HuggingFace 1.3B](https://huggingface.co/PKU-DS-LAB/Fairy-plus-minus-i-1.3B) · [ModelScope 700M](https://modelscope.cn/models/PKUDSLAB1806/Fairy-plus-minus-i-700M) · [ModelScope 1.3B](https://modelscope.cn/models/PKUDSLAB1806/Fairy-plus-minus-i-1.3B)



## Contents

1. [System Requirements](#1-system-requirements)
2. [Installation](#2-installation)
3. [Demo](#3-demo)
4. [Instructions for Use](#4-instructions-for-use)
5. [Introduction](#5-introduction)
6. [Results](#6-results)
7. [License](#7-license)



## Repository Structure

```text
model/          iFairy model and config (HuggingFace-compatible, trust_remote_code)
train/          Training script and accelerate / DeepSpeed config
eval/           Perplexity and zero-shot task evaluation
data_process/   RedPajama-Data-1T download, tokenization and sampling
requirements.txt        Pinned dependencies
```



# 1. System Requirements



## 1.1 Operating system

- Linux x86_64. Tested on **Ubuntu 22.04.5 LTS (Jammy)**.
- macOS and Windows are not supported, because `flash-attn` requires Linux + NVIDIA CUDA.



## 1.2 Hardware

- An **NVIDIA GPU with Ampere or newer architecture** (e.g. A100, A10, RTX 30xx/40xx, H800) is required for all training and evaluation scripts, since they use FlashAttention-2 and bf16. CPU-only machines are not supported.
- **Evaluation** runs on a single consumer GPU with at least 12 GB memory. It was tested on an NVIDIA RTX 4090 (24 GB); peak GPU memory with the commands in this README:

  | Model | `eval_ppl.py` | `eval_task.py` (`--batch_size 8`) |
  | ----- | ------------- | --------------------------------- |
  | 700M  | ~4.7 GB       | ~7.6 GB                           |
  | 1.3B  | ~7.0 GB       | ~10.6 GB                          |

- **Training** of the paper models used 32 × NVIDIA H800 GPUs. Fewer GPUs work as well; gradient accumulation is increased automatically to keep the global batch size at 512.
- **Data preparation** for the full ~100B-token dataset needs TB-scale disk space and a multi-core CPU (the commands below use 64 workers).



## 1.3 Software dependencies


| Software             | Tested version                                |
| -------------------- | --------------------------------------------- |
| Python               | 3.12.9                                        |
| CUDA (system)        | 12.9                                          |
| CUDA (PyTorch build) | 12.6                                          |
| torch                | 2.7.1                                         |
| transformers         | 4.52.4                                        |
| accelerate           | 1.7.0                                         |
| deepspeed            | 0.17.0                                        |
| flash-attn           | 2.8.1                                         |
| tokenizers           | 0.21.4                                        |
| safetensors          | 0.5.3                                         |
| datasets             | 2.19.0                                        |
| huggingface-hub      | 0.36.0                                        |
| numpy                | 2.2.6                                         |
| pyarrow              | 20.0.0                                        |
| zstandard            | 0.23.0                                        |
| requests             | 2.32.3                                        |
| lm-eval              | **0.3.0** (newer 0.4.x is **not** compatible) |
| glog                 | 0.3.1                                         |
| tqdm                 | 4.67.1                                        |
| swanlab              | 0.6.4 (optional training logging)             |
| tensorboard          | 2.19.0 (optional training logging)            |


These dependencies are listed in `requirements.txt`.

# 2. Installation



## 2.1 Steps

```bash
git clone https://github.com/PKU-LLM-DS-LAB/Fairy-plus-minus-i.git
cd Fairy-plus-minus-i

conda create -n ifairy python=3.12.9 -y
conda activate ifairy

# 1. PyTorch with CUDA 12.6
pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu126

# 2. Other dependencies
pip install -r requirements.txt

# 3. FlashAttention-2 (must be installed after torch)
pip install flash-attn==2.8.1 --no-build-isolation
```

Additional setup:

- **Llama-2 tokenizer access (training and data preparation only).** `train/train.py` and `data_process/tokenize_redpajama.py` use the tokenizer of `meta-llama/Llama-2-7b-hf`, which is a gated repository. Request access on its HuggingFace page, then run `huggingface-cli login`. Evaluation of the released models does not need this, since they ship their own tokenizer.
- **Logging (training only, optional).** Training logs are disabled by default. To log to [SwanLab](https://swanlab.cn), run `swanlab login` and pass `--report_to swanlab` to `train/train.py` (add `--swanlab_workspace NAME` to log to an organization workspace instead of your personal one). `--report_to tensorboard` writes TensorBoard logs to `./out/logs`.



## 2.2 Typical install time

About 10–20 minutes on a normal desktop with a broadband connection, mostly spent downloading PyTorch and the CUDA libraries (~3–4 GB). If no pre-built `flash-attn` wheel matches your environment it is compiled from source, which can take 30–60+ minutes (set `MAX_JOBS` to limit memory use).

# 3. Demo

The demo evaluates the released iFairy-700M model on small public datasets and reproduces the paper's 700M perplexity and two of its zero-shot accuracies. All demo data is downloaded automatically on first run:


| Data                | Source                                                  | Used by        |
| ------------------- | ------------------------------------------------------- | -------------- |
| WikiText2 test set  | `wikitext` / `wikitext-2-raw-v1` on HuggingFace         | `eval_ppl.py`  |
| C4 validation shard | `allenai/c4`, `en/c4-validation.00000-of-00008.json.gz` | `eval_ppl.py`  |
| PIQA, ARC-Easy      | downloaded by `lm-eval` 0.3.0                           | `eval_task.py` |




## 3.1 Instructions

```bash
# Perplexity on WikiText2 and C4
python eval/eval_ppl.py \
  --hf_path PKU-DS-LAB/Fairy-plus-minus-i-700M \
  --seqlen 2048 \
  --device cuda:0

# Zero-shot accuracy on PIQA and ARC-Easy
python eval/eval_task.py \
  --hf_path PKU-DS-LAB/Fairy-plus-minus-i-700M \
  --tasks piqa,arc_easy \
  --batch_size 8 \
  --num_fewshot 0 \
  --ctx_size 2048 \
  --device cuda:0
```



## 3.2 Expected output

`eval_ppl.py` prints the perplexity of each dataset and their average, matching the paper (WikiText2 10.45, C4 11.81, Avg 11.13) up to rounding. Last lines of the output on an RTX 4090:

```text
wikitext2 PPL: 10.458255236486018
c4 PPL: 11.812140858029887
[10.458255236486018, 11.812140858029887]
Avg PPL: 11.135198047257951
```

`eval_task.py` prints an `lm-eval` results table. The paper reports `acc` (ARC-Easy 53.45%, PIQA 68.01%). Output on an RTX 4090:

```text
|  Task  |Version| Metric |Value |   |Stderr|
|--------|------:|--------|-----:|---|-----:|
|arc_easy|      0|acc     |0.5362|±  |0.0102|
|        |       |acc_norm|0.4815|±  |0.0103|
|piqa    |      0|acc     |0.6812|±  |0.0109|
|        |       |acc_norm|0.6757|±  |0.0109|
```

Differences of a few tenths of a point from the paper (here +0.17 and +0.11) are expected; they come from the GPU model, batch size and library versions, and are well within one standard error.

## 3.3 Expected run time

On 1 × NVIDIA RTX 4090:


| Step                                                   | Time                              |
| ------------------------------------------------------ | --------------------------------- |
| First-time download of the model (3.1 GB) and datasets | ~10 min, depends on network speed |
| `eval_ppl.py`                                          | ~40 min                           |
| `eval_task.py` (PIQA + ARC-Easy)                       | ~5 min                            |




# 4. Instructions for Use



## 4.1 Prepare the training data

The training data is ~100B tokens randomly sampled from RedPajama-Data-1T, packed into 2048-token sequences. The scripts in `data_process/` build it in three steps:

```bash
# 1. Download raw files (resumable; use --max_files_per_subset N for a quick test, --url_base for a mirror)
python data_process/download_redpajama.py --output_dir RAW_DIR

# 2. Tokenize every document as [BOS] + tokens + [EOS] with the Llama-2 tokenizer
python data_process/tokenize_redpajama.py --raw_dir RAW_DIR --output_dir TOKEN_DIR --num_workers 64

# 3. Sample tokens from each subset in proportion to its size and pack into 2048-token blocks
python data_process/sample_redpajama.py --token_dir TOKEN_DIR --output_dir SAMPLE_DIR --num_workers 64
```

- Subsets: arxiv, c4, common_crawl, github, stackexchange, wikipedia (`book` is no longer distributed upstream).
- Each subset's budget is `target_tokens * subset_tokens / total_tokens`. Inside a subset, documents are drawn uniformly at random without replacement. The drawn documents are streamed through a buffer that is cut into 2048-token blocks; any leftover shorter than 2048 tokens carries over to the next documents, and only the final leftover is dropped.
- `--target_tokens` defaults to 104,857,600,000, matching `MAX_TOKEN` in `train/train.py`. The sampled total is slightly larger.
- `--ratio_source official` uses the dataset-card token counts instead of the counts measured on the tokenized data.
- The result `SAMPLE_DIR/dataset` (column `input_ids`, length 2048) is the `DATAPATH` for training; per-subset statistics are written to `SAMPLE_DIR/stats.json`.

**Using your own data.** Put your documents as `.jsonl`, `.jsonl.zst`, `.jsonl.gz` or `.json.gz` files (one `{"text": ...}` object per line) under `RAW_DIR/<subset>/`, where `<subset>` is one of the names above (or add your own name to `SUBSETS` in `data_process/common.py`), then run steps 2 and 3. Alternatively, provide any HuggingFace dataset saved with `save_to_disk` that has an `input_ids` column of length 2048 tokenized with the Llama-2 tokenizer.

## 4.2 Train

To start distributed training from the project root, run:

```bash
accelerate launch \
  --config-file train/complexnet_config.yaml \
  --num_processes N \
  train/train.py \
  --dataset_path DATAPATH
```

- `train/complexnet_config.yaml` — accelerate config (DeepSpeed ZeRO-2, bf16).
- `N` — number of GPUs.
- `DATAPATH` — the dataset from Section 4.1, loaded with `datasets.load_from_disk()`.
- `--report_to` — optional, `none` (default), `swanlab` or `tensorboard`; see Section 2.1.

Run the command from the project root so that the `model` package can be imported.

Training settings are constants at the top of `train/train.py`; edit them directly to change the run:


| Setting             | Value           | Meaning                                                                                           |
| ------------------- | --------------- | ------------------------------------------------------------------------------------------------- |
| `GLOBAL_BATCH_SIZE` | 512             | sequences per optimizer step                                                                      |
| `PER_DEVICE_BS`     | 4               | sequences per GPU per forward pass; gradient accumulation = 512 / (N × 4)                         |
| `MAX_TOKEN`         | 104,857,600,000 | total training tokens; the number of steps is `MAX_TOKEN // (GLOBAL_BATCH_SIZE × 2048)` = 100,000 |
| learning rate       | 1.5e-3          | peak LR for 700M (1.2e-3 for 1.3B), 375 warmup steps, two-stage decay (scaled by 0.6666 in the second half) |
| weight decay        | 0.1 → 0         | set to 0 after half of training                                                                   |
| `TRAIN_NAME`        | `out`           | checkpoints go to `./out/results`, the final model to `./out/saved_model`                         |
| `RESUME`            | `True`          | resume from the last checkpoint in `./out/results`                                                |


For a quick test run, lower `MAX_TOKEN` (e.g. `MAX_TOKEN = 10 * GLOBAL_BATCH_SIZE * 2048` trains for 10 steps) and `GLOBAL_BATCH_SIZE`.

**Model size.** The default `ComplexNetConfig` (`hidden_size=1536`, `intermediate_size=4096`, `num_hidden_layers=24`, `num_attention_heads=16`) is the 700M model, trained with a peak learning rate of 1.5e-3 (1.0e-3 in the second stage). For the 1.3B model, set `learning_rate=1.2e-3` in `TrainingArguments` (0.8e-3 in the second stage) and change the config in `train/train.py` to:

```python
config = ComplexNetConfig(
    vocab_size=new_vocab_size,
    hidden_size=2048,
    intermediate_size=5460,
    num_hidden_layers=24,
    num_attention_heads=32,
    num_key_value_heads=32,
    attn_implementation="flash_attention_2",
)
```

The saved model in `./out/saved_model` can be passed directly as `--hf_path` to the evaluation scripts.

## 4.3 Evaluate



### Zero-shot tasks

```bash
python eval/eval_task.py \
  --seed 42 \
  --hf_path /path/to/model_or_repo \
  --batch_size 8 \
  --device cuda:0 \
  --tasks TASK \
  --num_fewshot 0 \
  --ctx_size 2048 \
  --output_path results/tasks.json
```

- `--hf_path` — a local directory saved by `model.save_pretrained()`, or a HuggingFace repo id such as `PKU-DS-LAB/Fairy-plus-minus-i-700M`.
- `--tasks` — comma-separated `lm-eval` 0.3.0 task names, e.g. `arc_easy,arc_challenge,hellaswag,boolq,openbookqa,piqa,winogrande`.
- `--ctx_size` — maximum context length used during evaluation.
- `--output_path` — optional; saves the full results as JSON.



### Perplexity

```bash
python eval/eval_ppl.py \
  --seed 42 \
  --hf_path /path/to/model_or_repo \
  --seqlen 2048 \
  --device cuda:0
```

Evaluates on the WikiText2 test set and the first C4 English validation shard (`en/c4-validation.00000-of-00008.json.gz`).

- `--seqlen` — maximum sequence length; documents are packed up to this length.
- `--seed` — random seed.



## 4.4 Reproducing the paper results


| Result                                  | How to reproduce                                                                                                                       |
| --------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------- |
| iFairy 700M / 1.3B, PPL                 | `eval_ppl.py` with `--hf_path PKU-DS-LAB/Fairy-plus-minus-i-700M` or `...-1.3B`                                                        |
| iFairy 700M / 1.3B, zero-shot           | `eval_task.py` with `--tasks arc_easy,arc_challenge,hellaswag,boolq,openbookqa,piqa,winogrande --num_fewshot 0 --batch_size 8`         |
| iFairy from scratch                     | Section 4.1 (full data), then Section 4.2 on 32 × H800 (700M: ~70 h, 1.3B: ~148 h, with the 1.3B config and learning rate above)       |
| iFairy° (full-precision complex model)  | Same complex architecture without weight quantization, trained with the same data (Section 4.1) and learning-rate schedule as iFairy. |
| BF16 LLaMA (our trained baseline)       | Real-valued LLaMA-style model of the same scale in BF16, trained with the same data and learning-rate schedule as iFairy.            |
| BitNet b1.58† (our trained baseline)    | BitNet b1.58 trained with the same data (Section 4.1) and learning-rate schedule as iFairy.                                           |
| Symmetric INT2 (our trained baseline)   | 700M only. Real-valued QAT with the 2-bit codebook {−2, −1, 0, +1}, same data and learning-rate schedule as iFairy.                  |
| BitNet b1.58★                           | numbers are taken from prior work                                                                                                      |


Evaluation time for the released models on 1 × RTX 4090:


| Model | `eval_ppl.py` | `eval_task.py` (all 7 tasks) |
| ----- | ------------- | ---------------------------- |
| 700M  | ~40 min       | ~44 min                      |
| 1.3B  | ~70 min       | ~55 min                      |




# 5. Introduction

The advent of Large Language Models (LLMs) has transformed artificial intelligence, achieving remarkable performance across a wide range of natural language tasks. However, this success is built upon massive model sizes, often reaching billions or trillions of parameters, which poses serious deployment challenges due to immense memory footprints and high computational costs. To democratize access to these powerful models, model compression has become a critical research area, with *quantization* emerging as a leading technique. Quantization methods are broadly categorized into Post-Training Quantization (PTQ) and Quantization-Aware Training (QAT). While PTQ offers simplicity, its performance often degrades sharply in extremely low-bit scenarios due to the model's lack of adaptation to quantized representations. In contrast, QAT integrates quantization into the training loop, allowing models to learn robust low-bit representations and maintain performance under aggressive compression. This advantage has motivated recent research into QAT-based strategies tailored for LLMs.

The pursuit of extremely low-bit quantization, particularly 2-bit quantization, has become a focal point in efforts to compress LLMs for efficient deployment. Existing approaches, such as BitNet and its successors, have demonstrated that it is possible to retain reasonable accuracy using ternary quantization schemes with just 1.58 bits per weight. However, the accuracy of any quantized model is fundamentally limited by the following equation:

$$\textbf{Accuracy}_\text{quant}=\textbf{Accuracy}_\text{full-precision}-\textbf{Error}_\text{quant}$$

All current quantization research focuses on minimizing quantization error on full-precision models (e.g., LLaMA), but the quantization error can never be zero. Therefore, full-precision accuracy becomes the **ceiling** for quantized accuracy. To date, no existing method has even attempted to surpass this ceiling.

In this work, we propose a fundamentally different perspective. Instead of solely focusing on reducing quantization error, we make the first attempt to raise the ceiling (the accuracy of the full-precision model), while still ensuring that the resulting model can be efficiently quantized to a 2-bit format. Our key insight is that if the full-precision model becomes more expressive and accurate, the final 2-bit quantized model can achieve higher accuracy as well. Building on this insight, we propose, for the first time, incorporating complex-valued neural architectures into LLMs. The complex number provides a richer representational space with additional phase information, thereby enhancing the expressiveness of linear transformations without increasing the parameter count. By systematically extending the Transformer architecture into the complex domain, we construct a full-precision complex-valued LLM with superior modeling capacity.

Building upon this complex-valued foundation, we further design a novel 2-bit quantization scheme tailored for complex weights. Specifically, we quantize each complex parameter to one of the **fourth roots of unity** {±1, ±i} in the complex plane. This approach, unlike real-valued quantization, exploits the full 2-bit representational capacity *without sacrificing symmetry or sparsity*, thereby eliminating the trade-offs that limit real-valued schemes. The resulting model, which we name iFairy, is perfectly storage-efficient and phase-aware by design. We propose a quantization function that learns to project full-precision complex weights onto the target set {±1, ±i} while preserving both magnitude and phase information. We implement this within our complex Transformer framework and evaluate its performance under the same storage and compute constraints as BitNet b1.58. Experiments show that iFairy achieves the lowest average perplexity and the highest mean zero-shot accuracy among the 2-bit models at both 700M and 1.3B. It also surpasses the BF16 LLaMA reference in mean zero-shot accuracy at both scales and in average perplexity at 1.3B.

Our contributions can be summarized as follows:

- We propose a new perspective on low-bit quantization: improving the accuracy of quantized models by raising the ceiling (the full-precision model).
- We design a complex-valued LLM architecture that leverages the representational benefits of the complex domain without increasing parameter storage.
- We design a 2-bit quantization scheme that maps complex weights to the 4th roots of unity {±1, ±i}, fully utilizing bit capacity while preserving key properties like symmetry and sparsity.
- Experimental results show that our quantized model outperforms the ceiling of existing 2-bit quantization approaches in terms of both PPL and downstream understanding tasks.



# 6. Results

**Table: Perplexity on WikiText2 and C4 validation sets (lower is better)**


| Size | Model          | Quant. | WikiText2 | C4        | Avg       |
| ---- | -------------- | ------ | --------- | --------- | --------- |
| 700M | BF16 LLaMA     | No     | 10.58     | 11.45     | 11.02     |
|      | iFairy°        | No     | 9.41      | 10.75     | 10.08     |
|      | Symmetric INT2 | Yes    | 11.22     | 12.68     | 11.95     |
|      | BitNet b1.58★  | Yes    | –         | –         | 12.87     |
|      | BitNet b1.58†  | Yes    | 10.81     | 12.21     | 11.51     |
|      | **iFairy**     | Yes    | **10.45** | **11.81** | **11.13** |
| 1.3B | BF16 LLaMA     | No     | 9.65      | 10.77     | 10.21     |
|      | iFairy°        | No     | 8.72      | 9.95      | 9.34      |
|      | BitNet b1.58★  | Yes    | –         | –         | 11.29     |
|      | BitNet b1.58†  | Yes    | 9.57      | 11.08     | 10.32     |
|      | **iFairy**     | Yes    | **9.25**  | **10.85** | **10.05** |


**Table: Zero-shot accuracy on commonsense reasoning tasks (%)**


| Size | Model          | Quant. | ARCe      | ARCc      | HS        | BQ        | OQ        | PQ        | WGe       | Avg.      |
| ---- | -------------- | ------ | --------- | --------- | --------- | --------- | --------- | --------- | --------- | --------- |
| 700M | BF16 LLaMA     | No     | 53.62     | 22.95     | 36.00     | 60.09     | 18.80     | 67.79     | 53.04     | 44.61     |
|      | iFairy°        | No     | 55.68     | 24.06     | 37.79     | 60.46     | 20.60     | 70.18     | 54.46     | 46.18     |
|      | Symmetric INT2 | Yes    | 50.72     | 21.25     | 33.62     | 59.08     | 19.60     | 66.38     | 53.04     | 43.38     |
|      | BitNet b1.58★  | Yes    | 51.80     | 21.40     | 35.10     | 58.20     | 20.00     | 68.10     | 55.20     | 44.26     |
|      | BitNet b1.58†  | Yes    | 51.77     | 22.44     | 35.30     | 58.50     | 20.80     | 65.94     | 54.85     | 44.23     |
|      | **iFairy**     | Yes    | **53.45** | **23.04** | **36.04** | **57.31** | **21.00** | **68.01** | **54.06** | **44.70** |
| 1.3B | BF16 LLaMA     | No     | 55.89     | 25.09     | 37.69     | 57.98     | 22.60     | 69.64     | 55.33     | 46.32     |
|      | iFairy°        | No     | 58.96     | 25.77     | 40.29     | 60.92     | 23.20     | 71.44     | 57.06     | 48.23     |
|      | BitNet b1.58★  | Yes    | 54.90     | 24.20     | 37.70     | 56.70     | 19.60     | 68.80     | 55.80     | 45.39     |
|      | BitNet b1.58†  | Yes    | 56.19     | 23.38     | 37.93     | 59.33     | 21.20     | 68.44     | 55.88     | 46.05     |
|      | **iFairy**     | Yes    | **56.65** | **24.66** | **38.69** | **59.60** | **22.20** | **69.80** | **54.06** | **46.52** |


★ reported in prior work; † trained by us; ° full-precision iFairy (not quantized). BF16 LLaMA and Symmetric INT2 (codebook {−2, −1, 0, +1}, 700M only) are trained by us on the same data with the same schedule. Bold marks our method.

HS = HellaSwag, BQ = BoolQ, OQ = OpenBookQA, PQ = PIQA, WGe = WinoGrande.

# 7. License

This project is released under the [Apache License 2.0](LICENSE).