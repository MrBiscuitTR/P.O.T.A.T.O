from diffusers import DiffusionPipeline, PipelineQuantizationConfig
import torch
import datetime as dt
import os
import re

model_id = "tensorart/stable-diffusion-3.5-medium-turbo"
SAVE_FOLDER = os.path.join(os.path.dirname(__file__), "generated_images")
os.makedirs(SAVE_FOLDER, exist_ok=True)

os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_METRICS_OFF"] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
os.environ["HF_HUB_DISABLE_METADATA"] = "1"
os.environ["HF_HUB_DISABLE_TOKENS"] = "1"

quant_config = PipelineQuantizationConfig(
    quant_backend="bitsandbytes_4bit",   # <-- required
    quant_kwargs={
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_compute_dtype": torch.float16,
        "bnb_4bit_use_double_quant": True,
    },
)

# =========================
# PROCESS LAYER (ADDED)
# =========================

import multiprocessing as mp
import signal
import atexit
import threading
import time

_worker_process = None
_task_queue = None
_result_queue = None


def _worker(task_queue, result_queue, parent_pid):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load the pipeline correctly
    pipeline = DiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,  # Load in FP16 if GPU supports it
        quantization_config=quant_config,  # Apply quantization config
        safety_checker=None,        # disables safety checker
        # variant="fp16",            # ensure FP16 version if available
        use_safetensors=True        # key flag for safetensors
    ).to(device)

    def parent_watcher():
        while True:
            if not psutil.pid_exists(parent_pid):
                break
            time.sleep(1)
        task_queue.put(None)

    try:
        import psutil
        threading.Thread(target=parent_watcher, daemon=True).start()
    except Exception:
        pass

    while True:
        task = task_queue.get()
        if task is None:
            break

        prompt, topic, save_folder, save_name = task

        # Use autocast only if using CUDA
        autocast_device = "cuda" if device == "cuda" else "cpu"
        with torch.autocast(autocast_device):
            image = pipeline(
                prompt=prompt,
                negative_prompt="low quality, low detail, bad anatomy, mutated, deformed, blurry, watermark",
                width=768,
                height=1024,
                num_inference_steps=25,
                guidance_scale=5.5
            ).images[0]

        now = dt.datetime.now()
        formatted_date = now.strftime('%Y%m%d_%H%M%S')

        if save_folder is None:
            save_folder = SAVE_FOLDER

        os.makedirs(save_folder, exist_ok=True)

        if save_name is None:
            cleaned_prompt = re.sub(r'[<>:"/\\|?*]', ' ', prompt)
            cleaned_prompt = cleaned_prompt.replace(' ', '_')
            save_name = f"d-{formatted_date}+t-{topic}+{cleaned_prompt[-35:]}.png"

        save_path = os.path.join(save_folder, save_name)
        image.save(save_path)

        result_queue.put(save_path)
        torch.cuda.empty_cache()

    del pipeline
    torch.cuda.empty_cache()


def _start_worker():
    global _worker_process, _task_queue, _result_queue

    if _worker_process is not None:
        return

    mp.set_start_method("spawn", force=True)

    _task_queue = mp.Queue()
    _result_queue = mp.Queue()

    _worker_process = mp.Process(
        target=_worker,
        args=(_task_queue, _result_queue, os.getpid()),
        daemon=False
    )
    _worker_process.start()


def stop():
    global _worker_process
    if _worker_process is not None:
        _task_queue.put(None)
        _worker_process.join(timeout=10)
        if _worker_process.is_alive():
            _worker_process.terminate()
        _worker_process = None


def _cleanup(*args):
    stop()


atexit.register(_cleanup)
signal.signal(signal.SIGINT, _cleanup)
signal.signal(signal.SIGTERM, _cleanup)

# =========================
# ORIGINAL GENERATE FUNCTION (WRAPPED)
# =========================

def generate_image(prompt, topic, save_folder=None, save_name=None):
    _start_worker()
    _task_queue.put((prompt, topic, save_folder, save_name))
    return _result_queue.get()


# =========================
# CLI ENTRY (UNCHANGED LOGIC)
# =========================

def main():
    print("Image generation script started. Press Ctrl+C to exit.")
    while True:
        try:
            prompt = str(input("Enter a prompt to generate an image (or 'quit' to exit): "))
            if prompt.lower() == 'quit':
                break
            topic = "testmaintopic"
            path = generate_image(prompt, topic)
            print(f"Saved: {path}")
        except KeyboardInterrupt:
            print("\nCleaning up. ..")
            break
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
    stop()


if __name__ == "__main__":
    main()
