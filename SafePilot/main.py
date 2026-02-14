import cv2
import mediapipe as mp
import numpy as np
import config
from utils import calculate_ear, calculate_mar
from alarm import play_beep, stop_beep, play_warning, stop_warning

mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(refine_landmarks=True)

cap = cv2.VideoCapture(0)

LEFT_EYE = [33,160,158,133,153,144]
RIGHT_EYE = [362,385,387,263,373,380]
MOUTH = [78,95,88,178,87,14,317,402,318,324,308]

eye_counter = 0
perclos_buffer = []
initial_nose_y = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    h, w, _ = frame.shape

    if results.multi_face_landmarks:
        face = results.multi_face_landmarks[0]

        landmarks = []
        for lm in face.landmark:
            x, y = int(lm.x * w), int(lm.y * h)
            landmarks.append((x, y))

        left_eye = [landmarks[i] for i in LEFT_EYE]
        right_eye = [landmarks[i] for i in RIGHT_EYE]
        mouth = [landmarks[i] for i in MOUTH]

        ear_left = calculate_ear(left_eye)
        ear_right = calculate_ear(right_eye)
        ear = (ear_left + ear_right) / 2.0

        mar = calculate_mar(mouth)

        # Draw eye dots
        eye_color = (0,255,0) if ear > config.EAR_THRESHOLD else (0,0,255)
        for (x,y) in left_eye + right_eye:
            cv2.circle(frame, (x,y), 2, eye_color, -1)

        # -------- Eye Detection --------
        if ear < config.EAR_THRESHOLD:
            eye_counter += 1
            perclos_buffer.append(1)
        else:
            eye_counter = 0
            perclos_buffer.append(0)
            stop_beep()

        if eye_counter >= config.EAR_CONSEC_FRAMES:
            stop_warning()
            play_beep()
            cv2.putText(frame, "EYES CLOSED!", (50,50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

        # -------- Yawning --------
        if mar > config.MAR_THRESHOLD:
            cv2.putText(frame, "YAWNING!", (50,90),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

        # -------- PERCLOS --------
        if len(perclos_buffer) > config.PERCLOS_WINDOW:
            perclos_buffer.pop(0)

        perclos = (sum(perclos_buffer) / len(perclos_buffer)) * 100
        cv2.putText(frame, f"PERCLOS: {round(perclos,1)}%",
                    (50,130), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2)

        # -------- Head Detection --------
        nose_y = landmarks[1][1]

        if initial_nose_y is None:
            initial_nose_y = nose_y

        head_diff = nose_y - initial_nose_y

        if head_diff > config.HEAD_DROP_THRESHOLD:
            stop_beep()
            play_warning()
            cv2.putText(frame, "HEAD DOWN!", (50,170),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        else:
            stop_warning()

    cv2.imshow("SafePilot", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
