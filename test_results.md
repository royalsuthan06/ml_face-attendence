# Attendify: System Baseline Test Results

This document consolidates the official, high-resolution baseline performance, latency, and resource metrics of the **Attendify Face Attendance System** on the benchmarked environment. 

---

## 1. System Overview & Environment

*   **Operating System**: `Windows 10 Home Single Language 24H2 (Build 26100)`
*   **Python Runtime**: `3.12.10`
*   **CPU**: `AMD Ryzen 5 6600H with Radeon Graphics`
    *   *Specifications*: 6 Physical Cores / 12 Logical Processors
    *   *Clock Speed*: Current: `3301 MHz` / Max: `3301 MHz`
*   **System Memory (RAM)**: `15.31 GB` Total (Available at test start: `3.36 GB`)
*   **Video Controllers**:
    *   `NVIDIA GeForce RTX 3050 6GB Laptop GPU (4.0 GB dedicated)`
    *   `AMD Radeon(TM) Graphics (0.5 GB dedicated)`
    *   `Parsec Virtual Display Adapter`

---

## 2. Telemetry Results

### A. Cold Start Latency
Measures the cold-start instantiation cost before warm steady-state execution.

*   **Library Import Overhead**: `0.00 ms`
*   **Engine Initialization**: `19.32 ms` (constructing `FaceDetector` and `RecognitionEngine`)
*   **First-Run Cold Inference**: `2694.55 ms` (initial loading of model weights and JIT compile)
*   **Total System Cold-Start Time**: **`2713.87 ms`**

---

### B. Warm AI Processing Latency (Steady-State ms)
Measured over **100 warm iterations** after discarding 15 warm-up cycles.
*   *Note: Real computation running on synthetic frames. Accuracy metric: `NOT MEASURED`.*

| Measurement Target | Minimum | Average | Median | P95 | P99 | Maximum |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Face Detection** (HOG) | `5.37 ms` | `6.79 ms` | `6.81 ms` | `7.25 ms` | `7.51 ms` | `7.78 ms` |
| **Embedding Generation** (ResNet) | `599.07 ms` | `661.49 ms` | `660.84 ms` | `681.85 ms` | `726.68 ms` | `735.73 ms` |
| **End-to-End Inference** | `654.63 ms` | `672.72 ms` | `669.26 ms` | `694.47 ms` | `730.81 ms` | `738.01 ms` |

---

### C. Database Lookup Scaling Latency
Measures the matching engine's $O(N)$ computational scaling against simulated student databases.

| Database Records (Size) | Minimum | Average | Median | P95 | P99 | Maximum |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **10 records** | `0.09 ms` | `0.10 ms` | `0.09 ms` | `0.11 ms` | `0.15 ms` | `0.16 ms` |
| **50 records** | `0.23 ms` | `0.40 ms` | `0.39 ms` | `0.48 ms` | `0.52 ms` | `0.55 ms` |
| **100 records** | `0.46 ms` | `0.78 ms` | `0.77 ms` | `0.89 ms` | `1.13 ms` | `1.30 ms` |
| **250 records** | `1.14 ms` | `1.89 ms` | `1.87 ms` | `2.08 ms` | `2.16 ms` | `2.20 ms` |
| **500 records** | `3.45 ms` | `3.87 ms` | `3.82 ms` | `4.18 ms` | `4.93 ms` | `5.64 ms` |
| **1000 records** | `7.34 ms` | `9.22 ms` | `7.95 ms` | `12.17 ms` | `12.85 ms` | `12.87 ms` |

---

### D. Throughput & Camera Characteristics
*   **AI Throughput (Inference FPS)**: **`1.49 frames/sec`**
*   **Camera Capture FPS**: `Unavailable` (not exposed by webcam driver capture thread)
*   **Frame Buffer Read Rate**: **`437.25 reads/sec`** (verifies high-efficiency frame transfer rates)

---

### E. System Resources Footprint
Resource consumption under simulated normal active application workloads.

| Metric | Idle (App Start) | Active Workload (Simulated Normal) |
| :--- | :---: | :---: |
| **RAM RSS Footprint** | `337.98 MB` | `321.37 MB` (Peak: `330.34 MB`) |
| **Process CPU Utilization** | `3.10 %` | Avg: `90.13 %` / Peak: `100.10 %` |
| **System CPU Utilization** | `N/A` | Avg: `25.91 %` / Peak: `36.60 %` |
| **GPU Utilization (current)** | `0.00 %` | `0.00 %` (*Not utilized by current CPU-only backend*) |
| **VRAM Allocation** | `0 MB` | `0 MB` |

---

## 3. Key Telemetry Insights

### 1. The Core Bottleneck
The ResNet-34 **Embedding Generation** is the primary performance bottleneck. At **`661.49 ms`** average, it consumes **98.3%** of the entire end-to-end processing time. Additionally, this calculation consumes **90.13%** of the executing CPU thread, causing high thermal overhead and making it unsuitable for lower-spec CPU devices without acceleration.

### 2. Primary Strengths
*   **Face Bounding Box Detection**: OpenCV's classifier is extremely lightweight, completing in just **`6.79 ms`** average.
*   **Database Search Scale**: Matching face vectors against 1000 database records takes only **`9.22 ms`** average. Array comparisons are highly efficient.
*   **Buffer Transfers**: Multi-threaded camera frame buffers read at **`437.25 reads/sec`**, preventing frame drops at the UI streaming level.
