import cv2

class FaceDetector:
    def __init__(self, model_path=None):
        # Using Haar Cascade for simplicity, can be swapped for HOG or CNN
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def detect_faces(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        return faces # Returns list of (x, y, w, h)
