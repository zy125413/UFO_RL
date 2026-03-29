import os
import math
import pickle
import random
import time
from argparse import ArgumentParser

import datasets
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from trl.data_utils import maybe_apply_chat_template
from transformers import AutoModelForCausalLM, AutoTokenizer


def extract_answer_from_dataset(text):
    """
    GSM8K style:
    rationale #### final_answer
    """
    if "####" not in text:
        return None, None
    parts = text.split("####")
    rationale = parts[0].strip()
    answer = parts[1].strip().replace(",", "")
    return rationale, answer


def hyper_parameters():
    parser = ArgumentParser(description="Compute UFO-RL confidence and score.")
    parser.add_argument("--model_dir", type=str, default="./Qwen2.5-7B-Instruct")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--output", type=str, default="./output")
    parser.add_argument("--dataset_dir", type=str, default="./gsm8k")
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_str):
    if device_str.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return device_str


def load_train_split(dataset_dir):
    try:
        ds = datasets.load_dataset(dataset_dir)
    except Exception:
        ds = datasets.load_from_disk(dataset_dir)

    if "train" not in ds:
        raise ValueError(f"No 'train' split found in dataset: {dataset_dir}")
    return ds["train"]


def build_prompt_dataset(datas, system_prompt):
    def make_conversation(example):
        prompt = []
        if system_prompt is not None:
            prompt.append({"role": "system", "content": system_prompt})
        prompt.append({"role": "user", "content": example["question"]})
        return {"prompt": prompt}

    return datas.map(make_conversation)


def compute_sample_confidence(model, input_ids, attention_mask, answer_lens):
    """
    Paper Eq.:
        Conf(x_i) = (1 / T) * sum_t log P(y_t | x_i, y_<t)

    Here:
    - input_ids already contains prompt + gold answer tokens
    - answers are right-aligned at the sequence end
    - we read log-prob of each gold answer token from the previous position logits
    """
    max_answer_len = int(torch.max(answer_lens).item())
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        logits_to_keep=max_answer_len + 1,
    )
    log_probs = torch.nn.functional.log_softmax(outputs.logits, dim=-1)

    batch_conf = []
    batch_token_logprobs = []

    for b in range(input_ids.size(0)):
        cur_len = int(answer_lens[b].item())
        token_logprobs = []

        for i in range(cur_len):
            token_id = input_ids[b, -cur_len + i].item()
            # predict token at position (-cur_len + i) using logits from previous position
            token_lp = log_probs[b, -(cur_len + 1) + i, token_id].item()
            token_logprobs.append(token_lp)

        conf_i = float(sum(token_logprobs) / len(token_logprobs))
        batch_conf.append(conf_i)
        batch_token_logprobs.append(token_logprobs)

    return batch_conf, batch_token_logprobs


if __name__ == "__main__":
    hps = hyper_parameters()
    set_seed(hps.seed)
    device = resolve_device(hps.device)

    datas = load_train_split(hps.dataset_dir)

    tokenizer = AutoTokenizer.from_pretrained(hps.model_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    system_prompt = (
        "A conversation between User and Assistant. The user asks a question, "
        "and the Assistant solves it. The assistant first thinks about the reasoning "
        "process in the mind and then provides the user with the answer. "
        "The reasoning process and answer are enclosed within <think> </think> and "
        "<answer> </answer> tags, respectively, i.e., "
        "<think> reasoning process here </think>\n"
        "<answer> answer here </answer>."
    )

    datas = build_prompt_dataset(datas, system_prompt)

    answers = []
    for s in datas["solution"]:
        _, ans = extract_answer_from_dataset(s)
        if ans is None:
            raise ValueError("Found sample without '####' in solution field.")
        answers.append(ans)

    input_texts = []
    answer_lens = []

    for i in range(len(datas)):
        d = {"prompt": list(datas[i]["prompt"])}
        d["prompt"].append({"role": "assistant", "content": "<answer> " + answers[i]})
        input_text = maybe_apply_chat_template(d, tokenizer)["prompt"]
        input_texts.append(input_text)

        ans_ids = tokenizer(
            answers[i],
            add_special_tokens=False,
            return_attention_mask=False,
        )["input_ids"]
        answer_lens.append(len(ans_ids))

    answer_lens = torch.tensor(answer_lens, dtype=torch.long)

    inputs = tokenizer(
        input_texts,
        return_tensors="pt",
        padding=True,
        padding_side="left",
        add_special_tokens=False,
    )

    dataset = TensorDataset(
        inputs["input_ids"],
        inputs["attention_mask"],
        answer_lens,
    )
    dataloader = DataLoader(dataset, batch_size=hps.batch_size, shuffle=False)

    model = AutoModelForCausalLM.from_pretrained(hps.model_dir)
    model = model.to(device)
    if device.startswith("cuda"):
        model = model.half()
    model.eval()

    all_conf = []
    all_token_logprobs = []

    with torch.no_grad():
        begin = time.time()
        for batch in tqdm(dataloader):
            input_ids, attention_mask, batch_answer_lens = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            batch_answer_lens = batch_answer_lens.to(device)

            batch_conf, batch_token_logprobs = compute_sample_confidence(
                model=model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                answer_lens=batch_answer_lens,
            )
            all_conf.extend(batch_conf)
            all_token_logprobs.extend(batch_token_logprobs)
        end = time.time()

    print(f"Elapsed time: {end - begin:.2f}s")

    # Paper Sec 4.3:
    # s_i = exp(Conf(x_i))
    all_s = [math.exp(c) for c in all_conf]

    # mu is mean confidence score of candidate dataset
    mu = float(sum(all_s) / len(all_s))

    # Score(s_i) = 1 - (s_i - mu)^2
    answer_score = [1.0 - (s - mu) ** 2 for s in all_s]

    if not os.path.exists(hps.output):
        os.makedirs(hps.output)

    combined_conf = []
    combined_s = []
    combined_score = []

    for i in range(len(datas)):
        record = dict(datas[i])
        record["gold_answer"] = answers[i]

        combined_conf.append(
            {
                "sample": record,
                "token_logprobs": all_token_logprobs[i],
                "confidence": all_conf[i],   # Conf(x_i)
            }
        )
        combined_s.append(
            {
                "sample": record,
                "s": all_s[i],               # s_i = exp(Conf(x_i))
            }
        )
        combined_score.append(
            {
                "sample": record,
                "score": answer_score[i],    # 1 - (s_i - mu)^2
            }
        )

    with open(os.path.join(hps.output, "sample_confidence.pkl"), "wb") as f:
        pickle.dump(combined_conf, f)

    with open(os.path.join(hps.output, "sample_s.pkl"), "wb") as f:
        pickle.dump(combined_s, f)

    with open(os.path.join(hps.output, "answer_score.pkl"), "wb") as f:
        pickle.dump(
            {
                "mu": mu,
                "scores": combined_score,
            },
            f,
        )

    # Optional: directly save top-10% selected samples for RL training
    top_k = max(1, int(0.1 * len(combined_score)))
    top_indices = sorted(
        range(len(answer_score)),
        key=lambda i: answer_score[i],
        reverse=True
    )[:top_k]

    selected_samples = [datas[i] for i in top_indices]
    with open(os.path.join(hps.output, "top10_selected_samples.pkl"), "wb") as f:
        pickle.dump(selected_samples, f)

    print(f"Saved results to: {hps.output}")
    print(f"Dataset size: {len(datas)}")
    print(f"Top-10% selected: {top_k}")
    print(f"mu = {mu:.6f}")
