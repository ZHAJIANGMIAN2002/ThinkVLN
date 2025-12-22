torchrun --nproc_per_node=4 thinkvln/model/openloop_eval.py \
    --model_path outputs/lora/train_2025-12-18-12-19-49 \
    --dataset_path data/cot_dataset/sft_dataset_clean.json \
    --batch_size 32