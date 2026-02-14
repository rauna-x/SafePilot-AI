import cv2
import mediapipe as mp
import time
from collections import deque
from config import *
from utils import calculate_ear, calculate_mar
from alarm import trigger_warning, trigger_critical, stop_alarm

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()

cap = cv2.VideoCapture(0)

LEFT_EYE = [33,160,158,133,153,144]
RIGHT_EYE = [362,385,387,263,373,380]
MOUTH = [78,81,13,311,308,402,14,178]

eye_closed_start = None
perclos_queue = deque()
nose_positions = deque(maxlen=10)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    current_time = time.time()

    if results.multi_face_landmarks:
        face = results.multi_face_landmarks[0]

        left_eye = [(int(face.landmark[i].x*w), int(face.landmark[i].y*h)) for i in LEFT_EYE]
        right_eye = [(int(face.landmark[i].x*w), int(face.landmark[i].y*h)) for i in RIGHT_EYE]
        mouth_pts = [(int(face.landmark[i].x*w), int(face.landmark[i].y*h)) for i in MOUTH]

        ear = (calculate_ear(left_eye) + calculate_ear(right_eye)) / 2.0
        mar = calculate_mar(mouth_pts)

        nose_y = int(face.landmark[1].y*h)
        nose_positions.append(nose_y)

        # ---- Eye Color ----
        if ear < EAR_THRESHOLD:
            color = (0,0,255)
            perclos_queue.append(1)

            if eye_closed_start is None:
                eye_closed_start = current_time

            elapsed = current_time - eye_closed_start

            if elapsed > CLOSURE_CRITICAL:
                cv2.putText(frame,"CRITICAL WAKE UP!",
                            (30,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),3)
                trigger_critical()

            elif elapsed > CLOSURE_WARNING:
                cv2.putText(frame,"WARNING DROWSY",
                            (30,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,165,255),2)
                trigger_warning()
        else:
            color = (0,255,0)
            eye_closed_start = None
            perclos_queue.append(0)
            stop_alarm()

        # Draw eyes
        for p in left_eye:
            cv2.circle(frame,p,3,color,-1)
        for p in right_eye:
            cv2.circle(frame,p,3,color,-1)

        # ---- PERCLOS ----
        while len(perclos_queue) > PERCLOS_WINDOW*30:
            perclos_queue.popleft()

        if len(perclos_queue)>0:
            perclos = sum(perclos_queue)/len(perclos_queue)
            if perclos > PERCLOS_THRESHOLD:
                cv2.putText(frame,"HIGH FATIGUE (PERCLOS)",
                            (30,90),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
                trigger_warning()

        # ---- Yawning ----
        if mar > MAR_THRESHOLD:
            cv2.putText(frame,"YAWNING DETECTED",
                        (30,130),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)
            trigger_warning()

        # ---- Head Nod ----
        if len(nose_positions)==10:
            if max(nose_positions)-min(nose_positions)>HEAD_NOD_THRESHOLD:
                cv2.putText(frame,"HEAD NOD DETECTED",
                            (30,170),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,255),2)
                trigger_warning()

    cv2.imshow("Advanced Driver Drowsiness System", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()