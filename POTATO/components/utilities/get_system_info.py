import GPUtil
from torch import cuda as cuda

def get_all_system_info():
    print("Gathering system information...")
    print("-" * 40)
    # Get CPU info
    try:
        import psutil
        import platform

        cpu_count_logical = psutil.cpu_count(logical=True)
        cpu_count_physical = psutil.cpu_count(logical=False)
        cpu_freq = psutil.cpu_freq()
        cpu_percent = psutil.cpu_percent(interval=1)
        cpu_stats = psutil.cpu_stats()
        cpu_times = psutil.cpu_times()
        cpu_model = platform.processor()
        print(f"CPU Model: {cpu_model}")
        print(f"Physical Cores: {cpu_count_physical}")
        print(f"Logical Cores: {cpu_count_logical}")
        if cpu_freq:
            print(f"CPU Frequency: {cpu_freq.current:.2f} MHz (min: {cpu_freq.min:.2f}, max: {cpu_freq.max:.2f})")
        print(f"CPU Usage: {cpu_percent}%")
        print(f"CPU Stats: ctx_switches={cpu_stats.ctx_switches}, interrupts={cpu_stats.interrupts}, soft_interrupts={cpu_stats.soft_interrupts}, syscalls={cpu_stats.syscalls}")
        print(f"CPU Times: user={cpu_times.user}, system={cpu_times.system}, idle={cpu_times.idle}")
    except ImportError:
        print("psutil module not found. Skipping CPU info.")

    # Get RAM info
    try:
        import psutil
        virtual_mem = psutil.virtual_memory()
        swap_mem = psutil.swap_memory()
        print(f"Total RAM: {virtual_mem.total / (1024 ** 3):.2f} GB")
        print(f"Available RAM: {virtual_mem.available / (1024 ** 3):.2f} GB")
        print(f"Used RAM: {virtual_mem.used / (1024 ** 3):.2f} GB")
        print(f"RAM Usage: {virtual_mem.percent}%")
        print(f"Swap Total: {swap_mem.total / (1024 ** 3):.2f} GB")
        print(f"Swap Used: {swap_mem.used / (1024 ** 3):.2f} GB")
        print(f"Swap Usage: {swap_mem.percent}%")
    except ImportError:
        print("psutil module not found. Skipping RAM info.")

    print("-" * 40)

def get_gpu_info():
    try:
        gpus = GPUtil.getGPUs()
        if not gpus:
            print("No GPU found.")
            return

        print(f"Total GPUs detected: {len(gpus)}")
        for gpu in gpus:
            print(f"ID: {gpu.id}, Name: {gpu.name}")
            print(f"  UUID: {gpu.uuid}")
            print(f"  Driver Version: {gpu.driver}")
            print(f"  Memory Total: {gpu.memoryTotal} MB")
            print(f"  Memory Used: {gpu.memoryUsed} MB")
            print(f"  Memory Free: {gpu.memoryFree} MB")
            print(f"  Load: {gpu.load * 100:.1f}%")
            print(f"  Temperature: {gpu.temperature} °C")
            # print(f"  Display Mode: {gpu.display_mode}")
            print(f"  Display Active: {gpu.display_active}")
            print("-" * 40)
    except Exception as e:
        print(f"Error retrieving GPU info: {e}")

def get_gpu_memory_usage():
    print("Checking GPU memory usage using PyTorch...")
    if cuda.is_available():
        num_devices = cuda.device_count()
        print(f"Total CUDA Devices: {num_devices}")
        for i in range(num_devices):
            cuda.set_device(i)
            device_name = cuda.get_device_name(i)
            total_mem = cuda.get_device_properties(i).total_memory
            allocated = cuda.memory_allocated(i)
            reserved = cuda.memory_reserved(i)
            print(f"Device {i}: {device_name}")
            print(f"  Total Memory: {total_mem / (1024 ** 3):.2f} GB")
            print(f"  Allocated Memory: {allocated / (1024 ** 3):.4f} GB")
            print(f"  Reserved (Cached) Memory: {reserved / (1024 ** 3):.4f} GB")
            print(f"  Memory Usage: {(allocated / total_mem) * 100:.2f}%")
            print("-" * 20)
    else:
        print("GPU is not available.")
    print("-"*40)
    print("GPU check complete.")

def json_get_instant_system_info():
    import platform
    system_info = {}

    # CPU info
    try:
        import psutil
        cpu_count_logical = psutil.cpu_count(logical=True)
        cpu_count_physical = psutil.cpu_count(logical=False)
        cpu_percent = psutil.cpu_percent(interval=0.5)
        cpu_model = platform.processor()
        system_info["cpu"] = {
            "model": cpu_model,
            "physical_cores": cpu_count_physical,
            "logical_cores": cpu_count_logical,
            "usage_percent": cpu_percent
        }
    except ImportError:
        system_info["cpu"] = "psutil not available"

    # RAM info
    try:
        import psutil
        virtual_mem = psutil.virtual_memory()
        system_info["ram"] = {
            "total_gb": round(virtual_mem.total / (1024 ** 3), 2),
            "available_gb": round(virtual_mem.available / (1024 ** 3), 2),
            "used_gb": round(virtual_mem.used / (1024 ** 3), 2),
            "usage_percent": virtual_mem.percent
        }
    except ImportError:
        system_info["ram"] = "psutil not available"

    # GPU info
    gpu_info_list = []
    try:
        gpus = GPUtil.getGPUs()
        for gpu in gpus:
            gpu_info = {
                "id": gpu.id,
                "name": gpu.name,
                "uuid": gpu.uuid,
                "driver": gpu.driver,
                "memoryTotal_MB": gpu.memoryTotal,
                "memoryUsed_MB": gpu.memoryUsed,
                "memoryFree_MB": gpu.memoryFree,
                "load_percent": gpu.load * 100,
                "temperature_C": gpu.temperature,
                # "display_mode": gpu.display_mode,
                "display_active": gpu.display_active
            }
            gpu_info_list.append(gpu_info)
    except Exception as e:
        gpu_info_list.append({"error": f"Error retrieving GPU info: {e}"})
    system_info["gpus"] = gpu_info_list

    return system_info

if __name__ == "__main__":
    get_all_system_info() #optional
    get_gpu_info() #this suffices instead of the below though.
    get_gpu_memory_usage() #reauired for details, can be skipped for now. takes long because pytorch initializes cuda

