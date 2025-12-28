import GPUtil

def get_gpu_info():
    try:
        gpus = GPUtil.getGPUs()
        if not gpus:
            print("No GPU found.")
            return
        
        for gpu in gpus:
            print(f"ID: {gpu.id}, Name: {gpu.name}")
            print(f"  Driver Version: {gpu.driver}")
            print(f"  Memory Total: {gpu.memoryTotal}MB")
            print(f"  Memory Used: {gpu.memoryUsed}MB")
            print(f"  Memory Free: {gpu.memoryFree}MB")
            print(f"  Load: {gpu.load * 100:.1f}%")
            print(f"  Temperature: {gpu.temperature} °C")
            print("-" * 40)
    except Exception as e:
        print(f"Error retrieving GPU info: {e}")

if __name__ == "__main__":
    get_gpu_info()
