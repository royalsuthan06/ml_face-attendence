# Attendify: System Baseline Benchmark Report

This document records the baseline performance, resource consumption, and telemetry metrics of the **Attendify Face Attendance System** on the current hardware.

We have automated these measurements using [`benchmark_performance.py`](file:///x:/PROject/ml_face-attendence%20c/benchmark_performance.py), which collects data on AI processing speed, webcam frame rates, CPU/RAM utilization, and hardware specs.

---

## 1. Baseline Hardware Configuration

*   **Operating System**: `Windows 10 Home Single Language 24H2 (Build 26100)`
*   **Python Version**: `3.12.10`
*   **CPU**: `AMD Ryzen 5 6600H with Radeon Graphics` (6 Physical Cores / 12 Logical Threads)
*   **CPU Frequency**: `Current: 3301 MHz, Max: 3301 MHz`
*   **Total RAM**: `15.31 GB` (Available at start: `3.36 GB`)
*   **Graphics Acceleration**: CPU-only execution (GPU details detected: `Parsec Virtual Display Adapter`, `NVIDIA GeForce RTX 3050 6GB Laptop GPU [4.0 GB]`, `AMD Radeon(TM) Graphics [0.5 GB]`)

---

## 2. Benchmark Case 1: Cold Start Latency

Measured initialization separately from steady-state warm inference.

| Phase | Duration | Description |
| :--- | :---: | :--- |
| **Library Import Overhead** | `0.00 ms` | Loading standard and third-party modules. |
| **Engine Instantiation** | `19.32 ms` | Initializing `FaceDetector` and `RecognitionEngine`. |
| **First-Run Cold Inference** | `2694.55 ms` | The cold startup overhead of loading the dlib models and JIT compiler setup. |
| **Total Cold Start Time** | **`2713.87 ms`** | **Total initial latency before warm inference is reached.** |

---

## 3. Benchmark Case 2: Warm AI Latency (Steady-State ms)

We measured the execution latency over **100 warm iterations** (after discarding 15 warm-up cycles).
*   *Note: Because no real face dataset is checked into the workspace, these are computational benchmarks run on a synthetic frame, ensuring full ResNet execution but using dummy crops.*
*   **Recognition Accuracy**: `NOT MEASURED`

| Operation | Minimum | Average | Median | P95 | P99 | Maximum |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Face Detection** (HOG) | `5.37 ms` | `6.79 ms` | `6.81 ms` | `7.25 ms` | `7.51 ms` | `7.78 ms` |
| **Embedding Generation** (ResNet) | `599.07 ms` | `661.49 ms` | `660.84 ms` | `681.85 ms` | `726.68 ms` | `735.73 ms` |
| **End-to-End Pipeline** | `654.63 ms` | `672.72 ms` | `669.26 ms` | `694.47 ms` | `730.81 ms` | `738.01 ms` |

---

## 4. Benchmark Case 3: Matching Scale Test (ms)

Matching latency scales linearly with database size $O(N)$. We tested matching scaling under various simulated database record sizes.

| Database Size | Minimum | Average | Median | P95 | P99 | Maximum |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **10 records** | `0.09 ms` | `0.10 ms` | `0.09 ms` | `0.11 ms` | `0.15 ms` | `0.16 ms` |
| **50 records** | `0.23 ms` | `0.40 ms` | `0.39 ms` | `0.48 ms` | `0.52 ms` | `0.55 ms` |
| **100 records** | `0.46 ms` | `0.78 ms` | `0.77 ms` | `0.89 ms` | `1.13 ms` | `1.30 ms` |
| **250 records** | `1.14 ms` | `1.89 ms` | `1.87 ms` | `2.08 ms` | `2.16 ms` | `2.20 ms` |
| **500 records** | `3.45 ms` | `3.87 ms` | `3.82 ms` | `4.18 ms` | `4.93 ms` | `5.64 ms` |
| **1000 records** | `7.34 ms` | `9.22 ms` | `7.95 ms` | `12.17 ms` | `12.85 ms` | `12.87 ms` |

---

## 5. Benchmark Case 4: Throughput & Camera

*   **AI Throughput**: **`1.49 frames/sec`** (calculated from E2E latency)
*   **Camera Capture FPS**: `Unavailable` (not supported by raw `CameraModule` telemetry)
*   **Frame Buffer Read Rate**: **`437.25 reads/sec`** (verifies that frame-buffer sharing between threads is extremely fast)

---

## 6. Benchmark Case 5: System Resource Footprint

Measurements taken under active simulated normal Attendify load (simulated frame updates with a 150ms sleep gap):

*   **Idle RAM (RSS)**: `337.98 MB`
*   **Active RAM (RSS)**: `321.37 MB`
*   **Peak RAM (RSS)**: `330.34 MB`
*   **Idle Process CPU**: `3.1%`
*   **Active Process CPU**: Avg: `90.13%`, Peak: `100.1%`
*   **Active System CPU**: Avg: `25.91%`, Peak: `36.6%`
*   **GPU Utilization**: `0% / not used by current backend`
*   **VRAM Allocation**: `0 MB`

---

## 7. Baseline Analysis & Interpretation

### Current Bottleneck
*   **Embedding Generation**: Takes **`661.49 ms`** average, accounting for **98.3%** of the entire end-to-end processing pipeline latency. This is because dlib's ResNet face projection runs on a single CPU thread without GPU hardware acceleration.
*   **CPU Overhead**: Spikes the process thread to **90.13%** CPU utilization during active recognition, leaving minimal headroom for low-spec CPU devices.

### Current Strengths
*   **Face Detection**: Extremely fast. OpenCV's Cascade/HOG boundaries take only **`6.79 ms`** average, meaning face localization is not a performance bottleneck.
*   **Frame buffer read rate**: At **`437.25 reads/sec`**, the multi-threaded queueing system provides latency-free access to camera frames.
*   **Matching scale**: Even at 1000 database records, Euclidean matching takes only **`9.22 ms`** average, meaning matching logic scales very well for database lookups.

### Current Unknowns
*   **Low-Light Behavior**: We have not yet measured how low-light camera environments affect detection rates.
*   **Accuracy Details**: False Acceptance Rate (FAR) and False Rejection Rate (FRR) under different lighting conditions/distances.

### Recommended Next Benchmark
*   A **low-light benchmark** to record genuine recognition failure points under standard vs. dim classroom environments.
*   A **multi-person latency scaling test** to determine how E2E latency changes when 2, 3, or more faces appear concurrently in the webcam feed.

---

## 8. How to Re-Run Benchmarks

```powershell
# Run the baseline performance benchmark
.venv/Scripts/python.exe benchmark_performance.py

# Optional: Run a 10-minute stability test
.venv/Scripts/python.exe benchmark_performance.py --stability 10
```
