
cuda=0

CUDA_VISIBLE_DEVICES=${cuda} python generate_alpaca.py --pretrained_model_path Nexusflow/Starling-LM-7B-beta

CUDA_VISIBLE_DEVICES=${cuda} python evaluate_reward.py --task_name alpaca --input_name Starling-LM-7B-beta
