import cv2
import numpy as np
import base64
from gaze_tracking import GazeTracking
from face_detector import get_face_detector, find_faces
from face_landmarks import get_landmark_model, detect_marks

# -----------------------------
# Load models once
# -----------------------------

gaze = GazeTracking()

face_model = get_face_detector()
landmark_model = get_landmark_model()

net = cv2.dnn.readNetFromCaffe(
    "models/MobileNetSSD_deploy.prototxt",
    "models/MobileNetSSD_deploy.caffemodel"
)

CLASSES = ["background","aeroplane","bicycle","bird","boat",
           "bottle","bus","car","cat","chair","cow","diningtable",
           "dog","horse","motorbike","person","pottedplant",
           "sheep","sofa","train","tvmonitor","cell phone"]


# -----------------------------
# Phone + Person detection
# -----------------------------

def detect_phone_person(frame):
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 0.007843, (300,300), 127.5)
    net.setInput(blob)
    detections = net.forward()

    person_count = 0
    phone = False

    for i in range(detections.shape[2]):
        confidence = detections[0,0,i,2]
        if confidence > 0.6:
            idx = int(detections[0,0,i,1])
            label = CLASSES[idx]

            if label == "person":
                person_count += 1
            if label == "cell phone":
                phone = True

    return person_count, phone


# -----------------------------
# Main frame processor
# -----------------------------

def get_frame(imgData):

    nparr = np.frombuffer(base64.b64decode(imgData), np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # ---- Detect phone & persons
    person_count, phone_found = detect_phone_person(frame)

    if person_count == 0:
        person_status = 1
    elif person_count > 1:
        person_status = 2
    else:
        person_status = 0

    mob_status = 1 if phone_found else 0

    # ---- Face + head pose
    faces = find_faces(frame, face_model)
    user_move1 = 0
    user_move2 = 0

    for face in faces:
        marks = detect_marks(frame, landmark_model, face)
        nose = marks[30]
        chin = marks[8]

        if chin[1] - nose[1] > 40:
            user_move1 = 2   # Down
        elif nose[1] - chin[1] > 40:
            user_move1 = 1   # Up

    # ---- Eye tracking
    gaze.refresh(frame)

    if gaze.is_blinking():
        eye_movements = 1
    elif gaze.is_right():
        eye_movements = 4
    elif gaze.is_left():
        eye_movements = 3
    elif gaze.is_center():
        eye_movements = 2
    else:
        eye_movements = 0

    # ---- Encode frame
    _, jpeg = cv2.imencode(".jpg", frame)
    jpg_as_text = base64.b64encode(jpeg)

    return {
        "jpg_as_text": jpg_as_text,
        "mob_status": mob_status,
        "person_status": person_status,
        "user_move1": user_move1,
        "user_move2": user_move2,
        "eye_movements": eye_movements
    }
