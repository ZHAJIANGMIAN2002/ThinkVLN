export CUDA_VISIBLE_DEVICES=0

python -m thinkvln.datagen.generation.watcher_rollout_generation --summary_full_path data/trajectory_data/R2R_back/summary_full.jsonl --habitat_config_path config/vln_r2r.yaml --model_path outputs/actor/full/run-3-2-seperate-2 --bundle_root results/watcher_rollout_train_full_v2 --num_pivots 4 --num_rollouts 1 --min_rollout_steps 6 --max_rollout_steps 12 --seed 0 --resume
