
cuda=0
alpha=0.5

sft_model_path=openchat/openchat-3.5-0106
dpo_model_path=Nexusflow/Starling-LM-7B-beta

model_name=Starling-LM-7B-beta-ExPO

CUDA_VISIBLE_DEVICES=${cuda} python extrapolate.py --sft_model_path ${sft_model_path} --dpo_model_path ${dpo_model_path} \
    --alpha ${alpha} --save_path ./checkpoints/${model_name}

CUDA_VISIBLE_DEVICES=${cuda} python generate_alpaca.py --pretrained_model_path ./checkpoints/${model_name}

CUDA_VISIBLE_DEVICES=${cuda} python evaluate_reward.py --task_name alpaca --input_name ${model_name}
