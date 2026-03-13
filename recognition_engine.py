import face_recognition
import numpy as np

class RecognitionEngine:
    def __init__(self, threshold=0.6):
        self.threshold = threshold

    def get_locations(self, frame, model='hog'):
        # Convert BGR (OpenCV) to RGB (face_recognition) — copy to prevent memory issues
        rgb_frame = frame[:, :, ::-1].copy()
        return face_recognition.face_locations(rgb_frame, model=model)

    def get_encodings(self, frame, face_locations=None):
        # Convert BGR (OpenCV) to RGB (face_recognition)
        rgb_frame = frame[:, :, ::-1].copy()
        
        try:
            if face_locations is not None:
                # Ensure locations are a list of tuples (dlib requires this exact format)
                safe_locations = [tuple(int(v) for v in loc) for loc in face_locations]
                encodings = face_recognition.face_encodings(rgb_frame, safe_locations)
            else:
                encodings = face_recognition.face_encodings(rgb_frame)
        except Exception as e:
            print(f"[ENGINE ERROR] face_encodings failed: {e}")
            return []
            
        return encodings

    def compare_faces(self, known_encodings, face_encoding):
        if not known_encodings or len(known_encodings) == 0:
            return None, 1.0
        
        # Ensure all encodings are proper numpy arrays
        valid_encodings = []
        valid_indices = []
        for i, enc in enumerate(known_encodings):
            arr = np.asarray(enc, dtype=np.float64)
            if arr.ndim == 1 and arr.shape[0] == 128:
                valid_encodings.append(arr)
                valid_indices.append(i)
        
        if not valid_encodings:
            return None, 1.0
            
        distances = face_recognition.face_distance(valid_encodings, face_encoding)
        min_distance = np.min(distances)
        if min_distance < self.threshold:
            local_index = np.argmin(distances)
            original_index = valid_indices[local_index]
            return original_index, min_distance
        return None, min_distance
