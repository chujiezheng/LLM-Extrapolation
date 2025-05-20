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
    parser.add_argument('--output_path', type=str, default='./outputs_ultrafeedback')
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
    dataset = load_dataset('HuggingFaceH4/ultrafeedback_binarized', split='test_gen', num_proc=50)
    all_messages = [e[:1] for e in dataset['messages']]

    os.makedirs(args.output_path, exist_ok=True)

    # only for remedy
    if os.path.exists(f"{args.output_path}/{fname}.jsonl"):
        logging.info(f"File {args.output_path}/{fname}.jsonl exists, skipping")
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
    
    prompts = []
    outputs = []
    lengths = []
    finish_reasons = []
    for messages, generation in zip(all_messages, generations):
        query = messages[-1]['content']
        output = [e.text for e in generation.outputs][0]
        length = [len(e.token_ids) for e in generation.outputs][0]
        reason = [e.finish_reason for e in generation.outputs][0]
        prompts.append(query)
        outputs.append(output)
        lengths.append(length)
        finish_reasons.append(reason)

    timediff = time.time() - start
    logging.info(f"time elapsed: {timediff}")

    with open(f"{args.output_path}/{fname}.jsonl", "w") as f:
        for prompt, output, length, reason in zip(prompts, outputs, lengths, finish_reasons):
            f.write(json.dumps({"prompt": prompt, "output": output,
                                'length': length, 'finish_reason': reason}) + "\n")

    logging_cuda_memory_usage()


if __name__ == "__main__":
    main()
