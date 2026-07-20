#!/bin/bash
# TTS + LLM deps, installed after the main env build
set -ex
source /workspace/soulx/env.sh
pip install "misaki[en]" espeakng-loader num2words spacy curated-transformers
python - <<'PY'
import os
os.environ.setdefault('HF_HOME', '/workspace/.hf')
from huggingface_hub import hf_hub_download
hf_hub_download('hexgrad/Kokoro-82M', 'kokoro-v1_0.pth')
hf_hub_download('hexgrad/Kokoro-82M', 'config.json')
hf_hub_download('hexgrad/Kokoro-82M', 'voices/af_heart.pt')
from transformers import AutoModelForCausalLM, AutoTokenizer
AutoTokenizer.from_pretrained('Qwen/Qwen2.5-1.5B-Instruct')
AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-1.5B-Instruct')
print('TTS_LLM_ASSETS_OK')
PY
echo POST_ENV_DONE
