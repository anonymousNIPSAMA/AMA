#/bin/bash

python3 -m construct_privacy_info.construct_menory 

models=(
  "qwen3"
  "gpt-4o-mini"
  "gemma-3-it"
  "qwen2.5-instruct-32b"
  "llama-3.3-instruct"
)


for model in "${models[@]}"; do
  echo "Processing model: $model"

  python3 -m construct_privacy_info.generate_privacy_parameter --llm_name "$model"

  python3 generate_tool_metadata.py --llm_name "$model" --attack_type target -t 0.95 --lambda_weight 0.5

  python3 generate_tool_metadata.py --llm_name "$model" --attack_type target -t 0.8 --lambda_weight 0.5

  echo "Finished model: $model"
  echo "-------------------------"
done


python3 scripts/run.py -c config/all.yml

