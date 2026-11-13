import os
import sys
import subprocess
import re

# =============================================================
# 1. Bootstrap prebuilt CUDA-enabled llama-cpp-python for ZeroGPU
# =============================================================
def bootstrap_llama_cpp():
    print("Bootstrapping llama-cpp-python with CUDA for ZeroGPU...")
    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi()
    files = api.list_repo_files("AIencoder/llama-cpp-wheels", repo_type="dataset")

    # Priority 1: CUDA 12.4 / 12.6 wheels for ZeroGPU NVIDIA A10G
    cuda_candidates = [
        p for p in files 
        if p.endswith(".whl") 
        and ("cuda12.4" in p or "cuda12.6" in p) 
        and "basic" in p 
        and "/0.3.36/" in p
    ]

    # Priority 2: CPU fallback
    cpu_candidates = [
        p for p in files 
        if p.endswith(".whl") 
        and "linux/x86_64/cpu/" in p 
        and "/0.3.36/" in p
    ]

    chosen = cuda_candidates[0] if cuda_candidates else (cpu_candidates[0] if cpu_candidates else None)
    if not chosen:
        raise RuntimeError("No compatible prebuilt llama-cpp-python wheel found.")

    print(f"Selected wheel for ZeroGPU: {chosen}")
    whl_path = hf_hub_download("AIencoder/llama-cpp-wheels", chosen, repo_type="dataset")
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "--force-reinstall", "--no-deps", whl_path
    ])
    print("llama-cpp-python successfully installed!")

try:
    from llama_cpp import Llama
    print("llama-cpp-python already available.")
except (ImportError, RuntimeError, OSError):
    bootstrap_llama_cpp()
    from llama_cpp import Llama

import spaces
from typing import List
import gradio as gr

# =============================================================
# 2. ZeroGPU Lazy Model Loader
# =============================================================
# In ZeroGPU, CUDA is ONLY active inside @spaces.GPU decorated functions.
# Loading Llama globally at startup fails because no GPU context exists yet.
MODEL_PATH = "Qwen3.5-0.8B.Q4_K_M.gguf"
llm = None

def get_llm():
    global llm
    if llm is None:
        print(f"Initializing {MODEL_PATH} on ZeroGPU...")
        try:
            # Attempt full GPU layer offload
            llm = Llama(
                model_path=MODEL_PATH,
                n_ctx=4096,
                n_gpu_layers=-1,
                verbose=False
            )
            print("Model loaded successfully with GPU offload!")
        except Exception as e:
            print(f"GPU offload warning: {e}. Falling back to CPU execution.")
            llm = Llama(
                model_path=MODEL_PATH,
                n_ctx=4096,
                n_threads=2,
                n_gpu_layers=0,
                verbose=False
            )
            print("Model loaded successfully in CPU mode.")
    return llm

SYSTEM_PROMPT = (
    "You are Sherpa, an AI driving assistant for the CommitCon ride-hailing "
    "repositioning platform in Bengaluru. Provide concise, direct, helpful "
    "recommendations to drivers about where to reposition for high demand, "
    "earnings, and low traffic delays."
)

# =============================================================
# 3. Inference with ZeroGPU Allocation
# =============================================================
@spaces.GPU(duration=30)
def gradio_chat(user_message, history):
    # Model is loaded/invoked strictly INSIDE @spaces.GPU when CUDA context is granted
    model = get_llm()

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if history:
        for item in history:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                u, b = item
                messages.append({"role": "user", "content": str(u)})
                if b:
                    messages.append({"role": "assistant", "content": str(b)})
            elif isinstance(item, dict):
                messages.append(item)

    messages.append({"role": "user", "content": user_message})

    response = model.create_chat_completion(
        messages=messages,
        temperature=0.7,
        max_tokens=256
    )
    return response["choices"][0]["message"]["content"]

# =============================================================
# 4. Gradio Interface (Exposing clean /chat API)
# =============================================================
demo = gr.ChatInterface(
    fn=gradio_chat,
    title="🧭 CommitCon Sherpa Driver Copilot",
    description="Fine-tuned Qwen 0.8B Q4_K_M on ZeroGPU — llama-cpp-python",
    api_name="chat"
)

if __name__ == "__main__":
    demo.launch()
