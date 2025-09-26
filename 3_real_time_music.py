import cv2
import mediapipe as mp
import numpy as np
import joblib
import pygame

# Load trained model
model = joblib.load("models/hand_class_model.pkl")
scaler = joblib.load("models/scaler.pkl")
label_encoder = joblib.load("models/label_encoder.pkl")

# Initialize pygame mixer
pygame.mixer.init()

# Song mapping
music_files = {
    "right_hand": "sounds/Hungry Cheetah.mp3",
    "left_hand": "sounds/OMG_Daddy.mp3"
}

current_song = None

def play_music(file):
    global current_song
    if current_song != file:  # only load if new song
        pygame.mixer.music.load(file)
        pygame.mixer.music.play(-1)  # loop forever
        current_song = file

def stop_music():
    global current_song
    pygame.mixer.music.stop()
    current_song = None

# MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extract features
            coords = []
            for lm in hand_landmarks.landmark:
                coords.extend([lm.x, lm.y, lm.z])

            # Predict gesture
            prediction = model.predict([coords])[0]
            cv2.putText(frame, prediction, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Music control
            if prediction == "right_hand":
                play_music(music_files["right_hand"])
            elif prediction == "left_hand":
                play_music(music_files["left_hand"])
            elif prediction == "none":
                stop_music()

    cv2.imshow("Gesture Music Player", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
pygame.mixer.quit()
