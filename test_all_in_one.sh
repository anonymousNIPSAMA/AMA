#/bin/bash

python3 -m construct_privacy_info.construct_menory 

python3 -m construct_privacy_info.generate_privacy_parameter --llm_name qwen3 --attack_type target

python3  generate_tool_metadata.py --llm_name qwen3 --attack_type target -t 0.95 --lambda_weight 0.5

python3 scripts/run.py -c config/qwen3_target.yml