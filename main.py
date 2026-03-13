import cv2
import numpy as np
from database_manager import DatabaseManager
from camera_module import CameraModule
from face_detector import FaceDetector
from recognition_engine import RecognitionEngine
from attendance_manager import AttendanceManager

def main():
    db_manager = DatabaseManager()
    camera = CameraModule()
    detector = FaceDetector()
    engine = RecognitionEngine(threshold=0.5) # Set threshold for recognition
    attendance_mgr = AttendanceManager(db_manager, cooldown_minutes=5)

    # Load known employees
    employees = db_manager.get_all_employees()
    known_ids = [emp[0] for emp in employees]
    known_names = [emp[1] for emp in employees]
    known_encodings = [emp[2] for emp in employees]

    if not camera.open():
        return

    print("Starting attendance system. Press 'q' to quit.")

    while True:
        frame = camera.get_frame()
        if frame is None:
            print("Error: Could not read frame.")
            break

        # Detect faces
        face_locations = detector.detect_faces(frame)
        
        # If faces are detected, recognize them
        if len(face_locations) > 0:
            # Get encodings for all detected faces
            face_encodings = engine.get_encodings(frame, face_locations)
            
            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                # top, right, bottom, left are from face_recognition compatible format?
                # No, face_detector returns (x, y, w, h)
                x, y, w, h = top, right, bottom, left # These are actually x, y, w, h
                
                index, distance = engine.compare_faces(known_encodings, face_encoding)
                
                name = "Unknown"
                color = (0, 0, 255) # Red for unknown
                
                if index is not None:
                    name = known_names[index]
                    emp_id = known_ids[index]
                    color = (0, 255, 0) # Green for known
                    
                    # Mark attendance
                    marked, message = attendance_mgr.mark_attendance(emp_id, name)
                    if marked:
                        print(message)
                        # Optional: Visual cue for attendance marked
                        cv2.putText(frame, "Marked!", (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Draw label and rectangle
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.rectangle(frame, (x, y + h - 35), (x + w, y + h), color, cv2.FILLED)
                cv2.putText(frame, f"{name} ({distance:.2f})", (x + 6, y + h - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Show the frame
        if not camera.show_frame(frame, "Attendance System"):
            break

    camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
