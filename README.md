# Attendify 🎓 - Intelligent Face Recognition Attendance System

Attendify is a premium, real-time machine learning web application built to automate attendance tracking using face recognition. By combining modern computer vision frameworks with a clean, responsive web dashboard, Attendify provides an seamless biometric scanning and registration workflow.

---

## 🚀 Key Features

*   **High-Fidelity Real-Time Face Recognition**: Processes video feeds with optimized Hog-based face detection and ResNet-34 descriptor matching.
*   **Dual Enrollment System**: Register new students either via live webcam capture (with face distance/alignment validation) or by uploading high-quality photos.
*   **Live Analytics Dashboard**: View total students, present, and absent telemetry in real-time. Includes recent logs and manual check-in options.
*   **Heartbeat Watchdog System**: Prevents webcam hardware lockups by releasing the camera lock automatically if the browser tab/page is closed.
*   **Smart Attendance Logic**: Configurable logging cooldown periods to prevent duplicate scans, with automated status tagging (e.g., "Present" or "Late" after 9:00 AM).
*   **Obfuscated Security Vault**: Auto-saves unrecognized/unknown faces as encrypted `.sys` files in a vault folder.
*   **CSV Reports**: Filter and export attendance history based on date ranges, academic year, or department.

---

## 🛠️ Technology Stack

*   **Backend**: Python, Flask (Web framework)
*   **Database**: SQLite3 (Serverless relational database storing metadata and serialized 128D vector embeddings)
*   **Computer Vision / ML**:
    *   `dlib` / `face_recognition` (Face landmarks, ResNet-34 vector projection)
    *   `opencv-python` (Frame capture, resize interpolation, JPEG stream encoding)
    *   `numpy` (High-performance array operations)

---

## 📂 Project Structure

```text
├── app.py                      # Flask web server, background workers, and routes
├── database_manager.py          # SQLite interface, schemas, and student CRUD
├── camera_module.py             # Thread-safe multi-threaded webcam interface
├── face_detector.py             # Haar Cascade boundary detection wrapper
├── recognition_engine.py        # Vector encoding and Euclidean matching engine
├── attendance_manager.py        # Cooldown management and late status validation
├── requirements.txt             # Locked Python packages configuration
├── click_to_run_in_windows.bat  # Automated Windows bootstrap and launch script
├── templates/
│   └── index.html               # Real-time web dashboard using Tailwind CSS
└── data/
    └── logs/
        └── attendance.db        # Attendance database (Git ignored)
```

---

## 💻 Quick Start (Windows)

The repository includes a bootstrap batch script that installs all build tools, compiles C++ dependencies (`dlib`), installs libraries, and launches the app.

1. Clone the repository.
2. Ensure you have **Python 3.12 or 3.13** installed and added to your system `PATH`.
3. Double-click the [`click_to_run_in_windows.bat`](file:///x:/PROject/ml_face-attendence%20c/click_to_run_in_windows.bat) file.
4. Once setup completes, the application dashboard will automatically launch at:
   [**http://127.0.0.1:5000**](http://127.0.0.1:5000)

---

## 📊 Biometric Logic Details

### 1. Vector Projection
When registering, the application detects the face and maps 68 facial landmarks. It translates these landmarks into a **128-dimensional floating-point vector (embedding)** using a deep ResNet model.

### 2. Comparison Metrics
To recognize a face, the system calculates the **Euclidean distance** ($L_2$ norm) between the live frame's face embedding and all registered student embeddings in the database:
$$d = \sqrt{\sum_{i=1}^{128} (x_i - y_i)^2}$$
- **Threshold**: Set to `0.6`. Any match with a distance $d < 0.6$ is recognized.
- **Conflict Resolution**: If multiple matches are found within the threshold, the system chooses the closest match (minimum distance).
- **Anti-Duplicate Check**: During registration, the system scans the camera/upload image against existing records. If the face matches an already registered student, enrollment is rejected to prevent duplicate profiles.
