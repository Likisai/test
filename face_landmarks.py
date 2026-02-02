import cv2
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh

class HeadPoseEstimator:
    def __init__(self):
        self.face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def get_pose(self, frame):
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            return None

        face = results.multi_face_landmarks[0]

        # Nose tip
        nose = face.landmark[1]

        # Normalize to [-1, 1]
        x = (nose.x - 0.5) * 2
        y = (nose.y - 0.5) * 2

        if x > 0.15:
            return "Right"
        if x < -0.15:
            return "Left"
        if y > 0.15:
            return "Down"
        if y < -0.15:
            return "Up"

        return "Center"


def get_landmark_model():
    return HeadPoseEstimator()
