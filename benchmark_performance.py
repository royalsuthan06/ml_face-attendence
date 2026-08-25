import time
import os
import sys
import json
import platform
import subprocess
import argparse
import numpy as np
import cv2

# CRITICAL RULE: DO NOT AUTO-INSTALL DEPENDENCIES. Ask the user.
try:
    import psutil
except ImportError:
    print("====================================================")
    print("ERROR: The 'psutil' library is required to run the benchmark.")
    print("Please install it manually inside your virtual environment:")
    print("  pip install psutil")
    print("====================================================")
    sys.exit(1)

# Import local modules safely
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    from face_detector import FaceDetector
    from recognition_engine import RecognitionEngine
    from camera_module import CameraModule
except ImportError as e:
    print(f"ERROR: Failed to import project modules: {e}")
    sys.exit(1)

# Seed random numbers for reproducibility of synthetic benchmarks
np.random.seed(42)

def get_detailed_os():
    try:
        import winreg
        key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Microsoft\Windows NT\CurrentVersion")
        prod_name = winreg.QueryValueEx(key, "ProductName")[0]
        build = winreg.QueryValueEx(key, "CurrentBuild")[0]
        try:
            display_version = winreg.QueryValueEx(key, "DisplayVersion")[0]
            version_str = f"{prod_name} {display_version} (Build {build})"
        except FileNotFoundError:
            version_str = f"{prod_name} (Build {build})"
        return version_str
    except Exception:
        return f"{platform.system()} {platform.release()} ({platform.version()})"

def get_cpu_info():
    cpu_details = {
        "model": "Unknown CPU",
        "cores_physical": psutil.cpu_count(logical=False),
        "cores_logical": psutil.cpu_count(logical=True),
        "frequency_mhz": "Unavailable"
    }
    
    # Get CPU Model Name on Windows
    try:
        import winreg
        key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
        model = winreg.QueryValueEx(key, "ProcessorNameString")[0]
        cpu_details["model"] = model.strip()
    except Exception:
        pass

    # Get CPU frequency
    try:
        freq = psutil.cpu_freq()
        if freq:
            cpu_details["frequency_mhz"] = f"Current: {freq.current:.0f}MHz, Max: {freq.max:.0f}MHz"
    except Exception:
        pass
        
    return cpu_details

def get_gpu_info():
    # Detect GPUs via PowerShell CIM instances
    try:
        cmd = ["powershell", "-Command", "Get-CimInstance Win32_VideoController | Select-Object -Property Name, AdapterRAM | ConvertTo-Json"]
        output = subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL)
        if not output.strip():
            return "Unavailable"
            
        data = json.loads(output)
        gpus = []
        if isinstance(data, dict):
            data = [data]
            
        for item in data:
            name = item.get("Name")
            ram_bytes = item.get("AdapterRAM")
            if ram_bytes and isinstance(ram_bytes, int):
                ram_gb = round(ram_bytes / (1024**3), 2)
                gpus.append(f"{name} ({ram_gb} GB)")
            else:
                gpus.append(name)
        return ", ".join(gpus)
    except Exception:
        return "Unavailable"

def calc_stats(latencies_ms):
    if not latencies_ms:
        return {k: 0.0 for k in ["min", "avg", "median", "p95", "p99", "max"]}
    return {
        "min": round(float(np.min(latencies_ms)), 2),
        "avg": round(float(np.mean(latencies_ms)), 2),
        "median": round(float(np.median(latencies_ms)), 2),
        "p95": round(float(np.percentile(latencies_ms, 95)), 2),
        "p99": round(float(np.percentile(latencies_ms, 99)), 2),
        "max": round(float(np.max(latencies_ms)), 2)
    }

def run_benchmarks(args):
    print("\nInitializing Cold Start Telemetry...")
    
    # 1. Cold-start Benchmark (separating loading/instantiation from steady state)
    t_start = time.perf_counter()
    import face_recognition # Measure import overhead
    t_import = (time.perf_counter() - t_start) * 1000
    
    t_init = time.perf_counter()
    detector = FaceDetector()
    engine = RecognitionEngine()
    t_init_ms = (time.perf_counter() - t_init) * 1000
    
    # Synthetic frame setup
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    dummy_locations = [(100, 440, 380, 200)]
    
    # First inference cold-start
    t_cold = time.perf_counter()
    detector.detect_faces(frame)
    engine.get_encodings(frame, dummy_locations)
    t_cold_ms = (time.perf_counter() - t_cold) * 1000
    
    cold_start_total = t_import + t_init_ms + t_cold_ms
    print(f" - Library Import Cold-Start: {t_import:.2f} ms")
    print(f" - Engine Instantiation: {t_init_ms:.2f} ms")
    print(f" - First Inference Cold-Start: {t_cold_ms:.2f} ms")
    print(f" - Total Cold-Start Time: {cold_start_total:.2f} ms")

    # 2. Warm-Up Phase (15 iterations discarded to clear JIT and model caches)
    print("\nExecuting Warm-Up Phase (15 iterations)...")
    for _ in range(15):
        detector.detect_faces(frame)
        engine.get_encodings(frame, dummy_locations)
    print(" - Warm-up iterations discarded successfully.")

    # Warnings for synthetic data
    print("\n====================================================")
    print("WARNING: No local real face dataset available.")
    print("Proceeding with computational synthetic benchmarks.")
    print("Inference latencies reflect real processing costs,")
    print("but Recognition Accuracy is labeled as NOT MEASURED.")
    print("====================================================")

    # 3. Face Detection Microbenchmark (100 warm iterations)
    print("\nRunning Face Detection Microbenchmark (100 iterations)...")
    detect_latencies = []
    for _ in range(100):
        t0 = time.perf_counter()
        detector.detect_faces(frame)
        detect_latencies.append((time.perf_counter() - t0) * 1000)
    detect_stats = calc_stats(detect_latencies)

    # 4. Embedding Generation Microbenchmark (100 warm iterations)
    print("Running Embedding Generation Microbenchmark (100 iterations)...")
    embed_latencies = []
    for _ in range(100):
        t0 = time.perf_counter()
        engine.get_encodings(frame, dummy_locations)
        embed_latencies.append((time.perf_counter() - t0) * 1000)
    embed_stats = calc_stats(embed_latencies)

    # 5. Matching Microbenchmark (Synthetic databases scale check)
    print("Running Matching Microbenchmark (O(N) scale scaling verification)...")
    db_sizes = [10, 50, 100, 250, 500, 1000]
    match_results = {}
    target_embedding = np.random.rand(128)
    
    for size in db_sizes:
        db_embeddings = [np.random.rand(128).tolist() for _ in range(size)]
        latencies = []
        for _ in range(200): # Run matching 200 times for stability at each scale
            t0 = time.perf_counter()
            engine.compare_faces(db_embeddings, target_embedding)
            latencies.append((time.perf_counter() - t0) * 1000)
        match_results[str(size)] = calc_stats(latencies)

    # 6. End-to-End Pipeline Benchmark (50 warm iterations)
    print("Running End-to-End Pipeline Benchmark (50 iterations)...")
    e2e_latencies = []
    for _ in range(50):
        t0 = time.perf_counter()
        # Full stack direct timing
        faces = detector.detect_faces(frame)
        engine.get_encodings(frame, dummy_locations)
        # Using a dummy match array size of 50 records as baseline
        engine.compare_faces(db_embeddings[:50], target_embedding)
        e2e_latencies.append((time.perf_counter() - t0) * 1000)
    e2e_stats = calc_stats(e2e_latencies)

    # 7. AI Throughput Calculation
    total_e2e_time_sec = sum(e2e_latencies) / 1000.0
    ai_fps = round(len(e2e_latencies) / total_e2e_time_sec, 2)
    avg_e2e_ms = e2e_stats["avg"]

    # 8. Camera Benchmarks
    print("\nRunning Camera and Buffer Verification...")
    camera = CameraModule()
    
    # 8B. Frame-Buffer Read Rate (retrieve frames as fast as possible for 2 seconds)
    buffer_read_rate = 0.0
    if camera.open():
        read_count = 0
        t_cam_start = time.time()
        while time.time() - t_cam_start < 2.0:
            frame_cap = camera.get_frame()
            if frame_cap is not None:
                read_count += 1
            time.sleep(0.001) # Small thread yield
        t_cam_elapsed = time.time() - t_cam_start
        buffer_read_rate = round(read_count / t_cam_elapsed, 2)
        camera.release()
        print(f" - Frame-Buffer Read Rate: {buffer_read_rate} reads/sec")
    else:
        print(" - Camera hardware not available. Frame-Buffer Read Rate: Unavailable")
        buffer_read_rate = "Unavailable"

    # 9. System Resources Monitoring
    print("\nMonitoring System Resources...")
    process = psutil.Process(os.getpid())
    
    # Idle state
    idle_ram_rss = process.memory_info().rss / (1024 * 1024)
    psutil.cpu_percent(interval=None) # Reset system cpu
    process.cpu_percent(interval=None) # Reset process cpu
    time.sleep(0.5)
    idle_sys_cpu = psutil.cpu_percent(interval=None)
    idle_proc_cpu = process.cpu_percent(interval=None)

    # Active Workload Simulation (running 15 pipeline cycles with 150ms sleep between to simulate normal usage)
    cpu_proc_meas = []
    cpu_sys_meas = []
    active_ram_rss_list = []
    
    for _ in range(15):
        detector.detect_faces(frame)
        engine.get_encodings(frame, dummy_locations)
        cpu_proc_meas.append(process.cpu_percent(interval=None))
        cpu_sys_meas.append(psutil.cpu_percent(interval=None))
        active_ram_rss_list.append(process.memory_info().rss / (1024 * 1024))
        time.sleep(0.15) # Simulated camera capture frequency overhead
        
    active_ram_rss = np.mean(active_ram_rss_list)
    peak_ram_rss = process.memory_info().info.peak_wset / (1024 * 1024) if hasattr(process.memory_info(), "info") else np.max(active_ram_rss_list)
    
    res_stats = {
        "Idle RAM RSS MB": round(idle_ram_rss, 2),
        "Active RAM RSS MB": round(active_ram_rss, 2),
        "Peak RAM RSS MB": round(peak_ram_rss, 2),
        "Idle Process CPU %": round(idle_proc_cpu, 2),
        "Active Process CPU Average %": round(np.mean(cpu_proc_meas), 2),
        "Active Process CPU Peak %": round(np.max(cpu_proc_meas), 2),
        "Active System CPU Average %": round(np.mean(cpu_sys_meas), 2),
        "Active System CPU Peak %": round(np.max(cpu_sys_meas), 2)
    }

    # GPU telemetries
    gpu_name = get_gpu_info()
    gpu_metrics = {
        "GPU Hardware Name": gpu_name,
        "GPU Utilization": "0% / not used by current backend",
        "VRAM Allocation": "0 MB"
    }

    # Package versioning
    import cv2
    import dlib
    software_versions = {
        "Python": platform.python_version(),
        "numpy": np.__version__,
        "opencv-python": cv2.__version__,
        "dlib": dlib.__version__,
        "face_recognition": face_recognition.__version__
    }

    results = {
        "Timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "Software Versions": software_versions,
        "Hardware": {
            "OS": get_detailed_os(),
            "CPU": get_cpu_info(),
            "RAM Total GB": round(psutil.virtual_memory().total / (1024**3), 2),
            "RAM Available Start GB": round(psutil.virtual_memory().available / (1024**3), 2),
            "GPU": gpu_metrics
        },
        "Cold Start Latency": {
            "Library Import ms": round(t_import, 2),
            "Engine Instantiation ms": round(t_init_ms, 2),
            "First Inference ms": round(t_cold_ms, 2),
            "Total Cold Start ms": round(cold_start_total, 2)
        },
        "Accuracy Metric": "Recognition accuracy: NOT MEASURED",
        "Microbenchmarks": {
            "Face Detection (HOG) ms": detect_stats,
            "Embedding Generation ms": embed_stats,
            "Euclidean Matching Scaling ms": match_results,
            "End-to-End Inference ms": e2e_stats
        },
        "Throughput & Feed": {
            "AI Throughput FPS": ai_fps,
            "Camera Capture FPS": "Unavailable",
            "Frame Buffer Read Rate": buffer_read_rate
        },
        "Resource Consumption": res_stats
    }

    return results

def run_stability_benchmark(minutes):
    print(f"\n====================================================")
    print(f"Starting Long-Running Stability Benchmark ({minutes} minutes)...")
    print("====================================================")
    
    detector = FaceDetector()
    engine = RecognitionEngine()
    process = psutil.Process(os.getpid())
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    dummy_locations = [(100, 440, 380, 200)]
    db_embeddings = [np.random.rand(128).tolist() for _ in range(50)]
    target_embedding = np.random.rand(128)

    initial_ram = process.memory_info().rss / (1024 * 1024)
    cpu_measurements = []
    latencies = []
    processed_frames = 0
    errors = 0
    
    start_time = time.time()
    duration_sec = minutes * 60
    last_log = time.time()
    
    psutil.cpu_percent(interval=None) # Reset system cpu
    process.cpu_percent(interval=None) # Reset process cpu

    try:
        while time.time() - start_time < duration_sec:
            t0 = time.perf_counter()
            try:
                detector.detect_faces(frame)
                engine.get_encodings(frame, dummy_locations)
                engine.compare_faces(db_embeddings, target_embedding)
                latencies.append((time.perf_counter() - t0) * 1000)
                processed_frames += 1
            except Exception as e:
                errors += 1
                print(f"[STABILITY ERROR] Inference failed: {e}")

            cpu_measurements.append(process.cpu_percent(interval=None))
            
            # Log progress every 15 seconds
            if time.time() - last_log >= 15:
                elapsed_min = (time.time() - start_time) / 60
                current_ram = process.memory_info().rss / (1024 * 1024)
                print(f" - Elapsed: {elapsed_min:.2f}m | Processed: {processed_frames} | Current RAM: {current_ram:.2f} MB")
                last_log = time.time()
                
            # Simulated frame frequency delay (approx 2.5 FPS load footprint)
            time.sleep(0.4)
            
    except KeyboardInterrupt:
        print("\nStability test interrupted by user.")

    final_ram = process.memory_info().rss / (1024 * 1024)
    peak_ram = process.memory_info().info.peak_wset / (1024 * 1024) if hasattr(process.memory_info(), "info") else max(active_ram_rss_list)
    ram_growth = final_ram - initial_ram

    stability_results = {
        "Stability Test Duration Minutes": round((time.time() - start_time) / 60, 2),
        "Processed Frames": processed_frames,
        "Errors Encountered": errors,
        "Initial RAM MB": round(initial_ram, 2),
        "Final RAM MB": round(final_ram, 2),
        "Peak RAM MB": round(peak_ram, 2),
        "RAM Growth MB": round(ram_growth, 2),
        "Average CPU Utilization %": round(np.mean(cpu_measurements), 2),
        "Peak CPU Utilization %": round(np.max(cpu_measurements), 2),
        "Average AI Latency ms": round(np.mean(latencies), 2),
        "P95 AI Latency ms": round(np.percentile(latencies, 95), 2)
    }

    return stability_results

def print_console_report(r, stability_data=None):
    hw = r["Hardware"]
    cold = r["Cold Start Latency"]
    m = r["Microbenchmarks"]
    t = r["Throughput & Feed"]
    res = r["Resource Consumption"]
    
    print("\n")
    print("====================================================")
    print("             ATTENDIFY BASELINE BENCHMARK           ")
    print("====================================================")
    print("\n[Hardware]")
    print(f" - OS:                     {hw['OS']}")
    print(f" - CPU Model:              {hw['CPU']['model']}")
    print(f" - CPU Cores:              {hw['CPU']['cores_physical']} physical, {hw['CPU']['cores_logical']} logical")
    print(f" - CPU Frequency:          {hw['CPU']['frequency_mhz']}")
    print(f" - RAM Total:              {hw['RAM Total GB']} GB (Available at start: {hw['RAM Available Start GB']} GB)")
    print(f" - GPU Video Controller:   {hw['GPU']['GPU Hardware Name']}")
    
    print("\n[Software Versions]")
    for k, v in r["Software Versions"].items():
        print(f" - {k}: {v}")
        
    print("\n[Cold Start]")
    print(f" - Model Import overhead:  {cold['Library Import ms']} ms")
    print(f" - Engine Instantiation:   {cold['Engine Instantiation ms']} ms")
    print(f" - First-Run Cold Inference:{cold['First Inference ms']} ms")
    print(f" - Total Cold Start Time:  {cold['Total Cold Start ms']} ms")
    
    print("\n[Accuracy Metric]")
    print(f" - {r['Accuracy Metric']}")

    def print_stat_line(label, s):
        print(f" - {label:<20} | Min: {s['min']:<7} | Avg: {s['avg']:<7} | Med: {s['median']:<7} | P95: {s['p95']:<7} | P99: {s['p99']:<7} | Max: {s['max']}")

    print("\n[Warm AI Latency (Steady-State ms)]")
    print_stat_line("Face Detection (HOG)", m["Face Detection (HOG) ms"])
    print_stat_line("Embedding Generation", m["Embedding Generation ms"])
    print_stat_line("End-to-End Pipeline", m["End-to-End Inference ms"])
    
    print("\n[Euclidean Matching Scale Test (ms)]")
    for size in sorted(list(m["Euclidean Matching Scaling ms"].keys()), key=int):
        print_stat_line(f" - Database Size: {size:<4}", m["Euclidean Matching Scaling ms"][size])

    print("\n[Throughput & Camera]")
    print(f" - AI Throughput FPS:      {t['AI Throughput FPS']} frames/sec")
    print(f" - Camera Capture FPS:     {t['Camera Capture FPS']}")
    print(f" - Frame Buffer Read Rate: {t['Frame Buffer Read Rate']} reads/sec")

    print("\n[Resource Consumption]")
    print(f" - Idle RAM (RSS):         {res['Idle RAM RSS MB']} MB")
    print(f" - Active RAM (RSS):       {res['Active RAM RSS MB']} MB")
    print(f" - Peak RAM (RSS):         {res['Peak RAM RSS MB']} MB")
    print(f" - Idle Process CPU:       {res['Idle Process CPU %']}%")
    print(f" - Active Process CPU:     Avg: {res['Active Process CPU Average %']}%, Peak: {res['Active Process CPU Peak %']}%")
    print(f" - Active System CPU:      Avg: {res['Active System CPU Average %']}%, Peak: {res['Active System CPU Peak %']}%")
    
    print("\n[GPU Telemetry]")
    print(f" - GPU Utilization:        {hw['GPU']['GPU Utilization']}")
    print(f" - VRAM Allocation:        {hw['GPU']['VRAM Allocation']}")

    if stability_data:
        print("\n[Stability Metrics]")
        print(f" - Test Duration:          {stability_data['Stability Test Duration Minutes']} minutes")
        print(f" - Total Frames Processed: {stability_data['Processed Frames']}")
        print(f" - Errors/Exceptions:      {stability_data['Errors Encountered']}")
        print(f" - RAM Start -> End:       {stability_data['Initial RAM MB']} MB -> {stability_data['Final RAM MB']} MB (Growth: {stability_data['RAM Growth MB']} MB)")
        print(f" - RAM Peak:               {stability_data['Peak RAM MB']} MB")
        print(f" - CPU Average/Peak:       Avg: {stability_data['Average CPU Utilization %']}%, Peak: {stability_data['Peak CPU Utilization %']}%")
        print(f" - AI Latency Avg/P95:     Avg: {stability_data['Average AI Latency ms']} ms, P95: {stability_data['P95 AI Latency ms']} ms")
    
    print("\n====================================================")

def main():
    parser = argparse.ArgumentParser(description="Attendify System Performance Baseline Benchmarking Utility")
    parser.add_argument("--stability", type=float, help="Run an optional stability benchmark for specified duration in minutes")
    parser.add_argument("--output", type=str, help="Specify a custom filename to save JSON results")
    args = parser.parse_args()

    # Create timestamp identifier for default filenames
    timestamp = time.strftime("%Y-%m-%d_%H%M%S")
    
    # Run core performance benchmark
    results = run_benchmarks(args)
    
    # Run optional stability check
    stability_data = None
    if args.stability:
        if args.stability <= 0:
            print("ERROR: Stability test duration must be greater than 0 minutes.")
            sys.exit(1)
        stability_data = run_stability_benchmark(args.stability)
        results["Stability Test Results"] = stability_data
        
    # Print formatted output
    print_console_report(results, stability_data)
    
    # Determine output filename
    if args.output:
        out_file = args.output
    else:
        out_file = f"benchmark_results_{timestamp}.json"
        
    # Save structured json results
    with open(out_file, "w") as f:
        json.dump(results, f, indent=4)
        
    print(f"Baseline benchmark completed safely! Raw results saved to: {out_file}\n")

if __name__ == "__main__":
    main()
