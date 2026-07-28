cd src/ui_r1
pip install -e ".[dev]"

# Addtional modules
pip install wandb==0.18.3
pip install tensorboardx
pip install qwen_vl_utils torchvision
pip install flash-attn --no-build-isolation

# vLLM support 
# pip install vllm==0.7.2

# fix transformers version
# pip install transformers==4.49.0
