import cv2
import numpy as np
from database_manager import DatabaseManager
from camera_module import CameraModule
from face_detector import FaceDetector
from recognition_engine import RecognitionEngine
import time

def register_employee(name):
    db_manager = DatabaseManager()
    camera = CameraModule()
    detector = FaceDetector()
    engine = RecognitionEngine()

    if not camera.open():
        return

    print(f"Registering {name}. Please look at the camera. Press 's' to capture, 'q' to quit.")
    
    while True:
        frame = camera.get_frame()
        if frame is None:
            break
            
        faces = detector.detect_faces(frame)
        
        # Draw rectangles for feedback
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
        cv2.putText(frame, "Press 's' to capture", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow("Register Employee", frame)
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('s') and len(faces) > 0:
            # Capture and encode
            # Use the first face detected
            face_locations = [faces[0]]
            encodings = engine.get_encodings(frame, face_locations)
            
            if len(encodings) > 0:
                encoding = encodings[0]
                emp_id = db_manager.add_employee(name, encoding)
                print(f"Employee {name} registered with ID: {emp_id}")
                break
            else:
                print("Could not generate encoding. Try again.")
        
        elif key == ord('q'):
            break
            
    camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    name = input("Enter employee name: ")
    if name:
        register_employee(name)
