# reference: https://medium.com/@geronimo7/llms-multi-gpu-inference-with-accelerate-5a8333e4c5db

from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from vllm import utils as vllm_utils
from vllm import LLM, SamplingParams
from datasets import load_dataset

import argparse
import torch
import numpy as np
import time
import json
import random
import os
from pathlib import Path
from tqdm import tqdm
from datetime import timedelta
from utils import logging_cuda_memory_usage
from functools import partial
from datasets import load_from_disk
import pandas as pd

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
    parser.add_argument('--pretrained_model_path', type=str, required=True)
    parser.add_argument('--output_path', type=str, default='./outputs_alpaca')
    args = parser.parse_args()

    # prepare toker
    model_name = args.model_name = args.pretrained_model_path.split('/')[-1]
    logging.info(f"model_name: {model_name}")
    toker = AutoTokenizer.from_pretrained(
        args.pretrained_model_path,
        trust_remote_code=True,
        use_fase=True if 'internlm' in model_name else False,
    )
    if toker.pad_token is None:
        toker.pad_token = toker.eos_token
    toker.padding_side = "left"

    fname = model_name
    dataset = load_dataset("tatsu-lab/alpaca_eval", name="alpaca_eval")["eval"]
    dataset_names = [e['dataset'] for e in dataset]
    instructions = [e['instruction'] for e in dataset]
    all_messages = [[{'role': 'user', 'content': e}] for e in instructions]

    os.makedirs(args.output_path, exist_ok=True)

    # only for remedy
    if os.path.exists(f"{args.output_path}/{fname}.json"):
        logging.info(f"File {args.output_path}/{fname}.json exists, skipping")
        #return

    sampling_params = SamplingParams(
        n=1,
        temperature=0.7, top_k=50, top_p=0.9,
        presence_penalty=0.1, frequency_penalty=0.1,
        seed=42,
        max_tokens=2048,
        stop_token_ids=None if 'internlm' not in model_name else [92542],
    )
    model = LLM(
        model=args.pretrained_model_path,
        tokenizer=args.pretrained_model_path,
        trust_remote_code=True,
        dtype='bfloat16',
        gpu_memory_utilization=0.9,
        seed=42,
        tensor_parallel_size=torch.cuda.device_count(),
    )

    # sync GPUs and start the timer
    logging_cuda_memory_usage()
    logging.info(f"Running")

    input_ids = [toker.apply_chat_template(messages, add_generation_prompt=True, tokenize=True) for messages in all_messages]
    start = time.time()
    generations = model.generate(sampling_params=sampling_params, prompt_token_ids=input_ids)
    
    outputs = []
    for messages, generation in zip(all_messages, generations):
        output = [e.text for e in generation.outputs][0]
        outputs.append(output)

    timediff = time.time() - start
    logging.info(f"time elapsed: {timediff}")

    results = [{"dataset": dataset_name, "instruction": instruction, "output": output, "generator": model_name} 
               for dataset_name, instruction, output in zip(dataset_names, instructions, outputs)]
    with open(f"{args.output_path}/{fname}.json", "w") as f:
        json.dump(results, f, indent=2)

    logging_cuda_memory_usage()


if __name__ == "__main__":
    main()
