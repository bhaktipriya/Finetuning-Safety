#!/bin/bash
#SBATCH --time=1:00:00
#SBATCH --gpus-per-node=1
#SBATCH --account=srai
#SBATCH --output=safety_eval_%j.log
#SBATCH --error=safety_eval_%j.err


# Activate your virtual environment if needed
mamba activate qi
# Run the Python script
python -u safety_evaluation/question_inference.py \
  --model_name ckpts/Llama-2-7b-chat-fp16 \
  --prompt_file safety_evaluation/data/harmful_behaviors.csv \
  --prompt_template_style base \
  --output_file safety_evaluation/question_output/harmful_behaviors_llama_7b_full.jsonl
