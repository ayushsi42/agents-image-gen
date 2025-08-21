
## Quick Start

### Install & Run
```bash
# 1. Setup environment
conda env create -f environment.yaml
conda activate t2i_copilot

# 2. Put the OpenAI Key into .env
OPENAI_API_KEY=<YOUR_KEY_HERE>

# 3. Install required packages
pip install diffusers

# 4. Run with default settings (GPT-4o-mini, automatic mode)
python main.py --benchmark_name cool_sample.txt
```

## System Modes

| Mode | Description | Command |
|------|-------------|---------|
| **Closed-source MLLM** (Default) & **Automatic** (Default) | GPT-4o-mini via OpenAI API | `python main.py` |
| **Open-source MLLM** | Mistral/Qwen via vLLM server | `python main.py --use_open_llm` |
| **Human-in-the-loop** | Interactive user feedback | `python main.py --human_in_the_loop` |

## 🔧 Open Source MLLM Setup

**Use separate conda environment to avoid conflicts**

```bash
# Terminal 1: Start vLLM server
# Step i) 
conda env create -f env_vllm.yaml
conda activate vllm

# Step ii) Start server
# Option 1: Mistral
vllm serve mistralai/Mistral-Small-3.1-24B-Instruct-2503 --tokenizer_mode mistral --config_format mistral --load_format mistral --tool-call-parser mistral --enable-auto-tool-choice --limit_mm_per_prompt 'image=10' --tensor-parallel-size 2

# Option 2: Qwen2.5-VL (not recommend 3B)
vllm serve Qwen/Qwen2.5-VL-7B-Instruct

# Terminal 2: Run main.py (default model: Mistral-Samll-3.1-24B-Instruct-2503)
conda activate t2i_copilot
python main.py --use_open_llm
```

## Usage Examples

### Basic Usage
*Default using GPU to load Qwen-Image and Qwen-Image-Edit models*
```bash
# GPT-4o-mini + Automatic mode
python main.py --benchmark_name GenAIBenchmark/genai_image_seed.json

# GPT-4o-mini + Human-in-the-loop
python main.py --human_in_the_loop --benchmark_name cool_sample.txt

# Open-source LLM + Human-in-the-loop
python main.py --use_open_llm --human_in_the_loop
```

### Advanced Usage
```bash
# Custom open-source model and port
python main.py --use_open_llm \
    --open_llm_model="Qwen/Qwen2.5-VL-7B-Instruct" \
    --open_llm_port=8080 \
    --human_in_the_loop
```

###  Key Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--benchmark_name` | Benchmark file path | `cool_sample.txt` |
| `--human_in_the_loop` | Enable interactive mode | `False` |
| `--use_open_llm` | Use open-source LLM | `False` |
| `--open_llm_model` | Open LLM model name | `mistralai/Mistral-Small-3.1-24B-Instruct-2503` |
| `--open_llm_port` | vLLM server port | `8000` |

## Troubleshooting


- **PyTorch Error**: `RuntimeError: operator torchvision::nms does not exist`
    ```bash
    conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.4 -c pytorch -c nvidia
    ```

- **vLLM Conflicts**: Use separate conda environments
    - vLLM server: `transformers>=4.51.1,<4.53.0`
    - Main.py: Different transformers version