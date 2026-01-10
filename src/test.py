from model_handler import ModelManager
import os
import psutil

def get_vram_usage():
    # Simple check for system memory; 
    # Metal VRAM on Apple Silicon is Unified Memory
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

print("--- Initializing ModelManager ---")
handler = ModelManager()
print(f"Initial Memory Usage: {get_vram_usage():.2f} MB")

# Replace with a path to a Qwen2 local directory
MODEL_PATH = "./models/qwen2.5-code-7b-8bit" 

if os.path.exists(MODEL_PATH):
    print(f"--- Loading Model from {MODEL_PATH} ---")
    result = handler.load_model("Qwen2.5-7B", MODEL_PATH)
    print(result)
    print(f"Memory Usage after load: {get_vram_usage():.2f} MB")

    print("--- Testing 'Hot Swapping' (Clearing VRAM) ---")
    # Loading the same model or a different path should trigger the 'None' assignment in Rust
    handler.load_model("Empty-Trigger", "/non/existent/path")
    print(f"Memory Usage after clearing: {get_vram_usage():.2f} MB")
else:
    print(f"Error: Please place a Qwen2 model in {MODEL_PATH}")