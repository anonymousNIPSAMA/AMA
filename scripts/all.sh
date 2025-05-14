#!/bin/bash

bash ./kill.sh

python3 ./scripts/agent_attack_new.py -c ./config/reason.yml
python3 ./scripts/agent_attack_new.py -c ./config/mcp.yml
python3 ./scripts/agent_attack_new.py -c ./config/defense.yml
python3 ./scripts/agent_attack_new.py -c ./config/target_new.yml
python3 ./scripts/agent_attack_new.py -c ./config/benchmark.yml
python3 ./scripts/agent_attack_new.py -c ./config/origin.yml
python3 ./scripts/agent_attack_new.py -c ./config/gpt.yml
python3 ./scripts/agent_attack_new.py -c ./config/gpt_defense.yml



