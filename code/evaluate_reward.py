import os
import pandas as pd
import numpy as np
import torch
import json
from tqdm import tqdm
from transformers import AutoTokenizer, pipeline
import argparse
from datasets import disable_caching
disable_caching()
from utils import ListDataset
from torch.distributed.pipeline.sync import Pipe

import warnings
import logging
logging.basicConfig(
    format="[%(asctime)s] [%(filename)s:%(lineno)d] %(message)s",
    level=logging.INFO,
)
warnings.simplefilter("ignore")


def main():
    logging.info(f'cuda is available {torch.cuda.is_available()}')
    logging.info(f'cuda device count {torch.cuda.device_count()}')
    logging.info(f'cuda device name {torch.cuda.get_device_name()}')

    parser = argparse.ArgumentParser()
    parser.add_argument('--task_name', type=str, required=True, choices=['ultrafeedback', 'alpaca'])
    parser.add_argument('--input_name', type=str, required=True)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained("weqweasdas/RM-Mistral-7B")
    rm_pipe = pipeline(
        "sentiment-analysis",
        model="weqweasdas/RM-Mistral-7B",
        device_map='auto',
        tokenizer=tokenizer,
        torch_dtype=torch.bfloat16,
    )
    # You may encounter a bfloat16 dtype bug of transformers >= 4.40
    # Refer to https://github.com/huggingface/transformers/pull/30996 to modify the source code

    pipe_kwargs = {
        "return_all_scores": True,
        "function_to_apply": "none",
        "batch_size": 8,
        "num_workers": 20,
    }

    os.makedirs(f'rewards_{args.task_name}', exist_ok=True)
    prompts = []
    outputs = []
    if args.task_name == "ultrafeedback":
        with open(f"outputs_{args.task_name}/{args.input_name}.jsonl", "r") as f:
            for line in f:
                data = json.loads(line)
                prompts.append(data["prompt"])
                outputs.append(data["output"])
    elif args.task_name == "alpaca":
        with open(f"outputs_{args.task_name}/{args.input_name}.json", "r") as f:
            data = json.load(f)
        for d in data:
            prompts.append(d["instruction"])
            outputs.append(d["output"])

    chats = [[{'role': 'user', 'content': str(prompt)}, {'role': 'assistant', 'content': str(output)},
              ] for prompt, output in zip(prompts, outputs)]
    texts = [tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=False).replace(tokenizer.bos_token, "") for chat in chats]

    logging.info(f"Running {args.input_name}")
    text_dataset = ListDataset(texts)
    pipe_outputs = [e for e in tqdm(rm_pipe(text_dataset, **pipe_kwargs), total=len(texts), dynamic_ncols=True)]
    logging.info(f"Done {args.input_name}")
    rewards = [output[0]["score"] for output in pipe_outputs]
    with open(f"rewards_{args.task_name}/{args.input_name}.jsonl", "w") as f:
        for reward in rewards:
            f.write(json.dumps(reward) + "\n")


if __name__ == '__main__':
    main()