"""
Central configuration for the Smart LLM Router.
All model definitions, ports, and serving parameters live here.
"""

MODELS = {
    "qwen-coder": {
        "name": "Qwen 2.5 Coder 3B",
        "model_id": "Qwen/Qwen2.5-Coder-3B-Instruct-AWQ",
        "port": 8001,
        "purpose": "Simple tasks, code generation, debugging, classifier",
        "quantization": "awq",
        "max_model_len": 4096,
        "gpu_memory_utilization": 0.40,
    },
    "llama-8b": {
        "name": "Llama 3.1 8B",
        "model_id": "hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4",
        "port": 8002,
        "purpose": "Complex reasoning, analysis, fallback",
        "quantization": "awq",
        "max_model_len": 2048,
        "gpu_memory_utilization": 0.55,
    },
}

MODEL_TIERS = ["qwen-coder", "llama-8b"]

VLLM_COMMON_ARGS = {
    "dtype": "float16",
    "enforce_eager": True,
}
