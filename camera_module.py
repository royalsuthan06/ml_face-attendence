import cv2
import threading
import time

class CameraModule:
    def __init__(self, camera_index=0):
        self.camera_index = camera_index
        self.cap = None
        self.frame = None
        self.running = False
        self.lock = threading.Lock()
        self.thread = None

    def open(self):
        if self.is_opened():
            return True
            
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            print(f"Error: Could not open camera {self.camera_index}")
            self.cap = None
            return False
            
        self.running = True
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()
        return True

    def _update(self):
        """Background thread to read frames into buffer."""
        while self.running:
            if self.cap is not None:
                ret, frame = self.cap.read()
                if ret:
                    with self.lock:
                        self.frame = frame
                else:
                    time.sleep(0.01)
            else:
                time.sleep(0.1)

    def is_opened(self):
        return self.cap is not None and self.cap.isOpened() and self.running

    def get_frame(self):
        with self.lock:
            if self.frame is None:
                return None
            return self.frame.copy()

    def release(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
        try:
            if self.cap is not None:
                self.cap.release()
        except Exception:
            pass
        finally:
            self.cap = None
            self.frame = None
            self.thread = None

    def show_frame(self, frame, window_name="Attendance System"):
        cv2.imshow(window_name, frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return False
        return True

