export MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet
MASTER_PORT=$((RANDOM % 101 + 20000))

CHECKPOINT="/mnt/swx/ThinkVLN/model_weights/streamvln"
echo "CHECKPOINT: ${CHECKPOINT}"

torchrun --nproc_per_node=1 --master_port=$MASTER_PORT streamvln/streamvln_eval.py --model_path /mnt/swx/ThinkVLN/model_weights/streamvln
