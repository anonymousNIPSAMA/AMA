
# AMA Codebase

We are currently refactoring the codebase and progressively uploading updates.

---

## 🚀 Setup Instructions

Follow the steps below to set up the environment and run the project.

### 1. Create a Python Environment

We recommend using `conda` for environment management:

```bash
conda create --name ama python=3.11
conda activate ama
````

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🧪 Running Tests with Xinference

To run the tests, you must launch a **Xinference** service with the correct model and settings.

### 1. Launch Xinference

We recommend deploying Xinference using Docker. For detailed instructions, please refer to the [official documentation](https://github.com/xorbitsai/inference).

Currently supported models include:
* `qwen3`
* `gemma-3-it-27b`
* `qwen2.5-instruct-32b`
* `llama-3.3-instruct`

Example command to launch `qwen3` with `vLLM` engine:

```bash
xinference launch \
  --model-name qwen3 \
  --model-uid qwen3 \
  --model-engine vllm \
  --size-in-billions 32 \
  --model-format pytorch \
  -r 8 \
  --gpu-memory-utilization 0.9 \
  --max-model-len 22784
```

### 2. Configure API Access

Update the configuration in `llm_utils/utils.py`:

```python
xinference_api_key = 'sk-xxxx'  # Replace with your actual API key
xinference_api_base = "http://xxx.xxx.xxx.xxx:9997/v1"  # Replace with your actual API endpoint
```

For GPT-based models, also modify:

```python
gpt_link = "xxxx"
gpt_key = "sk-xxxx"
```

---

## ▶️ Run the Test Script (Step-by-step)

Make the script executable and run it:

```bash
chmod +x test_all_in_one.sh
./test_all_in_one.sh
```

This script performs the following steps:

1. **Construct a privacy dataset**

   ```bash
   python3 -m construct_privacy_info.construct_menory
   ```

2. **Generate function parameters**

   ```bash
   python3 -m construct_privacy_info.generate_privacy_parameter --llm_name qwen3 --attack_type target
   ```

3. **Generate adversarial metadata**

   ```bash
   python3 generate_tool_metadata.py --llm_name qwen3 --attack_type target -t 0.95 --lambda_weight 0.5
   ```

4. **Run the main attack pipeline**

   ```bash
   python3 scripts/run.py -c config/qwen3_target.yml
   ```

---

## ⚔️ Trying Untargeted Attacks

To try **untargeted** attack settings, simply run:

```bash
python3 generate_tool_metadata.py --llm_name qwen3 --attack_type untarget -t 0.8 --lambda_weight 0.5
python3 scripts/run.py -c config/qwen3_untarget.yml
```

Logs can be inspected afterward to evaluate attack effectiveness.

---

## 🧩 Extended Evaluation (AMA + Injected Attack)

We provide a more comprehensive evaluation setup, combining:

* AMA attacks
* Injected prompt attacks
* With/without defense mechanisms

To run the **entire workflow (from data generation to evaluation)** in one step, use:

```bash
chmod +x run_all_in_one.sh
./run_all_in_one.sh
```

---

> This script will sequentially execute all major components including data construction, metadata generation, and evaluation with both target and untarget attacks.

---

## 📌 Notes

* The code is under **active development** — breaking changes may occur.
* Contributions, issues, and suggestions are welcome and appreciated.

---

## 📄 License

This code is licensed under the MIT License. 
