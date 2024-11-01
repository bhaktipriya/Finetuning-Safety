#!/bin/bash
#SBATCH --job-name=finetuning
#SBATCH --nodes=1
#SBATCH --time=1:00:00
#SBATCH --gpus-per-node=2
#SBATCH --account=srai


# Activate your environment
mamba activate qi  # Replace with your Conda environment

# Run the torchrun command
torchrun --nnodes 1 --nproc_per_node 2 finetuning.py \
  --batch_size_training 64 --lr 2e-5 \
  --gradient_accumulation_steps 1 --weight_decay 0 \
  --num_epochs 1 \
  --dataset dolly_dataset \
  --enable_fsdp \
  --model_name ckpts/Llama-2-7b-chat-fp16 --pure_bf16 \
  --dist_checkpoint_root_folder finetuned_models/ \
  --dist_checkpoint_folder dolly-7b-full
