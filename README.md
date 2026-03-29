# UFO-RL: Uncertainty-Focused Optimization for Efficient RL Data Selection

[![Paper: NeurIPS 2025](https://img.shields.io/badge/Paper-NeurIPS%202025-blue.svg)](https://neurips.cc/virtual/2025/loc/mexico-city/poster/115706)

This repository contains the official implementation of **UFO-RL: Uncertainty-Focused Optimization for Efficient Reinforcement Learning Data Selection**.

UFO-RL is a lightweight and scalable framework for efficient RL data selection in large language model post-training. Inspired by the Zone of Proximal Development (ZPD), it identifies the most informative training samples using a single-pass uncertainty estimate, avoiding expensive multi-sampling during data evaluation.

## Highlights

- **Efficient difficulty evaluation:** up to **185×** faster than multi-sample accuracy estimation
- **Efficient RL fine-tuning:** achieves comparable or better performance using only **10%** of the training data
- **Training efficiency:** reduces overall RL training time by up to **16×**
- **Simple scoring pipeline:** compute sample confidence, transform it into a fuzziness score, and keep the top 10% most informative samples

## Method

For each candidate sample \(x_i\), we compute the model confidence as the average log-probability of the target sequence:

\[
\mathrm{Conf}(x_i)=\frac{1}{T}\sum_{t=1}^{T}\log P(y_t \mid x_i, y_{<t})
\]

We then define:

\[
s_i=\exp(\mathrm{Conf}(x_i))
\]

Let \(\mu\) be the mean of \(s_i\) over the candidate dataset. The final fuzziness score is:

\[
\mathrm{Score}(s_i)=1-(s_i-\mu)^2
\]

Samples with higher scores are closer to the model's current uncertainty center and are selected for RL training. In the paper, the top **10%** samples ranked by this score are used as the final training subset.

## Repository Structure

```text
.
├── compute_ufo_scores.py
├── README.md
````

## Input Format

The current script assumes a GSM8K-style dataset with at least the following fields:

* `question`
* `solution`

and expects `solution` to contain:

```text
rationale #### final_answer
```

By default, the script extracts the text after `####` as the gold answer target.

## Usage

Run the scoring script with:

```bash
python compute_ufo_scores.py \
  --model_dir /path/to/your/model \
  --dataset_dir /path/to/your/dataset \
  --output ./ufo_output \
  --batch_size 8 \
  --device cuda
```

### Arguments

* `--model_dir`: path to the pretrained model
* `--dataset_dir`: path to the dataset
* `--output`: directory for saved outputs
* `--batch_size`: batch size for scoring
* `--device`: device string such as `cuda`, `cuda:0`, or `cpu`
* `--seed`: random seed

## Outputs

The script saves the following files:

* `sample_confidence.pkl`
  Per-sample confidence values `Conf(x_i)` and token-level log-probabilities

* `sample_s.pkl`
  Per-sample transformed values `s_i = exp(Conf(x_i))`

* `answer_score.pkl`
  Final fuzziness scores `1 - (s_i - mu)^2` and the dataset mean `mu`

* `top10_selected_samples.pkl`
  Top 10% highest-scoring samples selected for RL training

## Example Workflow

### Step 1: Score the candidate data

```bash
python compute_ufo_scores.py \
  --model_dir ./Qwen2.5-7B-Instruct \
  --dataset_dir ./gsm8k \
  --output ./ufo_output \
  --batch_size 8 \
  --device cuda
```

### Step 2: Convert selected samples for RL training

After obtaining `top10_selected_samples.pkl`, convert the selected subset into the dataset format expected by your RL training pipeline.

### Step 3: Train with `open-r1`

UFO-RL is designed to work naturally with GRPO-style RL fine-tuning pipelines such as `open-r1`.

Example:

```bash
ACCELERATE_LOG_LEVEL=info accelerate launch \
  --config_file recipes/accelerate_configs/zero2.yaml \
  src/open_r1/grpo.py \
  --model_name_or_path /path/to/your/model \
  --dataset_name /path/to/your/converted_top10_dataset \
  --learning_rate 1e-6 \
  --max_prompt_length 512 \
  --max_completion_length 1024 \
  --temperature 1.0 \
  --num_generations 7 \
  --output_dir ./outputs/ufo_grpo
```

## Results

In the paper, UFO-RL demonstrates strong efficiency-performance trade-offs across multiple models and benchmarks.

For example, when trained on **GSM8K** with **Qwen2.5-7B**, the full-data baseline achieves **91.88** accuracy on the GSM8K test set, while **UFO-RL reaches 92.03 using only 10% of the data**.

For difficulty evaluation on the same model, single-pass confidence scoring takes **175s**, compared with **6827s** for multi-sample accuracy estimation, corresponding to a **39× speedup**.

## Notes

* This implementation follows the paper-level scoring pipeline:

  * sample-level confidence
  * exponential transform
  * fuzziness score centered around the dataset mean
* The current script scores the **final answer tokens** extracted after `####`
* If your experiment uses the **full solution sequence** instead of only the final answer, adjust the target construction accordingly

## Safety Notice

* Only load models and datasets from trusted sources
* Be careful when opening `.pkl` files from untrusted sources, since Python `pickle` is not a secure interchange format

## Citation

If you find this code useful, please cite our NeurIPS 2025 paper:

```bibtex
@inproceedings{zhao2025uforl,
  title={UFO-RL: Uncertainty-Focused Optimization for Efficient Reinforcement Learning Data Selection},
  author={Yang Zhao and Kai Xiong and Xiao Ding and Li Du and Yangou Ouyang and Zhouhao Sun and Jiannan Guan and Wenbin Zhang and Bin Liu and Dong Hu and Bing Qin and Ting Liu},
  booktitle={Advances in Neural Information Processing Systems},
  year={2025},
  note={Poster},
  url={https://openreview.net/forum?id=sH0ZwzDJZn}
}
```

## Links

* **NeurIPS 2025 Poster Page:** [https://neurips.cc/virtual/2025/loc/mexico-city/poster/115706](https://neurips.cc/virtual/2025/loc/mexico-city/poster/115706)
