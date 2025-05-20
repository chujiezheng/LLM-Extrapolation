# LLM-Extrapolation

Official repository for **ACL 2025** paper "[Model Extrapolation Expedites Alignment](https://arxiv.org/abs/2404.16792)"

If you find this repository useful or our work is related to your research, please kindly cite it:
```latex
@inproceedings{
  llm-extrapolation,
  title={Model Extrapolation Expedites Alignment},
  author={Chujie Zheng and Ziqi Wang and Heng Ji and Minlie Huang and Nanyun Peng},
  booktitle={The 63rd Annual Meeting of the Association for Computational Linguistics
},
  year={2025}
}
```


## Models

We have uploaded the trained checkpoints and extrapolated models on 🤗 HuggingFace.

For the extrapolated models applied to open-source models, see [this 🤗 HuggingFace collection](https://huggingface.co/collections/chujiezheng/model-extrapolation-expedites-alignment-662b69fbe7850e722e10ff70).

For the `zephyr` checkpoints trained from `zephyr-7b-sft-full` in our controlled experiments, see [this 🤗 HuggingFace collection](https://huggingface.co/collections/chujiezheng/model-checkpoints-in-the-expo-paper-662b00fde58d277c81fb5bfb).

## Implementation of ExPO

The implementation of ExPO is extremely simple. You can refer to the code `code/extrapolate.py` (setting alpha to 0.3 or 0.5 is usually good).

## Experimental Results

You can find the raw outputs of the standardized benchmarks AlpacaEval 2.0 (`results_alpaca`), MT-Bench (`results_mtbench`), and Open LLM Leaderboard (`results_lmeval`). For Open LLM Leaderboard, you can find the scores of the non-existing models from the [official leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard).

We have also uploaded the AlpacaEval 2.0 evaluation results to the [official leaderboard](https://tatsu-lab.github.io/alpaca_eval/). You can find the detailed inference hyperparameters in their repository for reproduction.

## Inference and Evaluation Code

The inference code includes `code/generate_ultrafeedback.py` and `code/generate_alpaca.py`. The script `code/scripts/Starling-LM-7B-beta_extra.sh` shows:

* Do model extrapolation (ExPO) with a DPO/RLHF and its initial SFT checkpoints
* Use a HuggingFace model to generate responses on UltraFeedback or AlpacaEval 2.0. The outputs will be saved to `outputs_ultrafeedback` or `outputs_alpaca`
* Score the outputs using the reward model. The reward scores will be saved to `rewards_ultrafeedback` or `rewards_alpaca`

For the part of evaluation on standardized benchmarks:

* To run the official AlpacaEval 2.0 evaluation, follow https://github.com/tatsu-lab/alpaca_eval?tab=readme-ov-file#evaluating-a-model
* To run the official MT-Bench evaluation, follow https://github.com/lm-sys/FastChat/tree/main/fastchat/llm_judge (you can host the local vllm server to speed up inference)
* To run the official Open LLM Leaderboard evaluation, follow https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard (About -> REPRODUCIBILITY)
