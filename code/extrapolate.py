import torch
import argparse
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sft_model_path', type=str, required=True)
    parser.add_argument('--dpo_model_path', type=str, required=True)
    parser.add_argument('--alpha', type=float, required=True)
    parser.add_argument('--save_path', type=str, required=True)
    args = parser.parse_args()
    
    sft_model = AutoModelForCausalLM.from_pretrained(
        args.sft_model_path,
        torch_dtype=torch.bfloat16,
        device_map='auto',
        trust_remote_code=True,
    )
    sft_model.generation_config.do_sample = True
    dpo_model = AutoModelForCausalLM.from_pretrained(
        args.dpo_model_path,
        torch_dtype=torch.bfloat16,
        device_map='auto',
        trust_remote_code=True,
    )
    dpo_model.generation_config.do_sample = True

    assert len(sft_model.state_dict()) == len(dpo_model.state_dict())
    total = len(dpo_model.state_dict())

    for name, dpo_model_param in tqdm(dpo_model.named_parameters(), total=total):
        sft_model_param = sft_model.state_dict()[name]
        dpo_model_param.data = dpo_model_param.data + args.alpha * (dpo_model_param.data - sft_model_param.data)

    dpo_model.save_pretrained(args.save_path)
    toker = AutoTokenizer.from_pretrained(args.dpo_model_path, trust_remote_code=True)
    toker.save_pretrained(args.save_path)


if __name__ == '__main__':
    main()
