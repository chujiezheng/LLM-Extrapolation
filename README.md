# LLM-Extrapolation

Official repository for paper "[Weak-to-Strong Extrapolation Expedites Alignment](https://arxiv.org/abs/2404.16792)" [[tweet]](https://twitter.com/ChujieZheng/status/1783911895088632175)

If you find this repository useful or our work is related to your research, please kindly cite it:
```
@article{
  llm-extrapolation,
  title={Weak-to-Strong Extrapolation Expedites Alignment},
  author={Chujie Zheng and Ziqi Wang and Heng Ji and Minlie Huang and Nanyun Peng},
  journal={arXiv preprint arXiv:2404.16792},
  year={2024}
}
```

## News

**[05/2024]** Our uploaded **ExPO**-enhanced LLMs [[paper]](https://arxiv.org/abs/2404.16792) have received **<font color="red">10K+ downloads in 2 weeks</font>** on [HuggingFace](https://huggingface.co/collections/chujiezheng/weak-to-strong-extrapolation-expedites-alignment-662b69fbe7850e722e10ff70)!

## Models

We have uploaded the trained checkpoints and extrapolated models on HuggingFace.

For the extrapolated models applied to open-source models, see [this HuggingFace collection](https://huggingface.co/collections/chujiezheng/weak-to-strong-extrapolation-expedites-alignment-662b69fbe7850e722e10ff70).

For the `zephyr` checkpoints trained from `zephyr-7b-sft-full` in our controlled experiments, see [this HuggingFace collection](https://huggingface.co/collections/chujiezheng/model-checkpoints-in-the-expo-paper-662b00fde58d277c81fb5bfb).

## Implementation of ExPO

The implementation of ExPO is extremely simple. You can refer to the code `extrapolate.py` (setting alpha to 0.3 or 0.5 may be good).

## Experimental Results

We have uploaded the AlpacaEval 2.0 evaluation results to the [official leaderboard](https://tatsu-lab.github.io/alpaca_eval/). You can find the detailed inference hyperparameters in their repository for reproduction.

## Inference and Evaluation Code

TBD.
