# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 10:19:37 2022

@author: USER
"""

import os
import numpy as np
import cv2
import pyttsx3
from playsound import playsound
from keras.preprocessing.image import img_to_array
from keras.models import load_model
from mtcnn import MTCNN

# Python text to speech parameters
engine = pyttsx3.init()
voices = engine.getProperty('voices')
engine.setProperty('rate', 150)
engine.setProperty('volume', 1)
engine.setProperty('voice', voices[1].id)

# Audio Files
alarm = os.path.join(os.path.dirname(__file__), "alarm.wav")
drowsy = os.path.join(os.path.dirname(__file__), "drowsy.mp3")

# Load the pre-trained facial landmark model
face_detector = MTCNN()

# Setup the Thresholds
EAR_THRESHOLD = 0.24
MOR_THRESHOLD = 0.6
NLR_LOWER_THRESHOLD = 0.75
NLR_UPPER_THRESHOLD = 1.3
MOE_THRESHOLD = 2.4

# Initialize parameters
blink_consec_count = 0
yawn_consec_count = 0
drowsy_blink_count = 0
yawn_count = 0

# Function to calculate EAR (Eye Aspect Ratio)
def EAR(drivereye):
    print(drivereye[3])
    point1 = np.linalg.norm(drivereye[1] - drivereye[3])
    point2 = np.linalg.norm(drivereye[2] - drivereye[3])
    distance = np.linalg.norm(drivereye[0] - drivereye[3])
    ear_aspect_ratio = (point1 + point2) / (2.0 * distance)
    return ear_aspect_ratio

# Function to calculate MOR (Mouth Aspect Ratio)
def MOR(drivermouth):
    point = np.linalg.norm(drivermouth[0] - drivermouth[3])
    point1 = np.linalg.norm(drivermouth[2] - drivermouth[3])
    point2 = np.linalg.norm(drivermouth[3] - drivermouth[3])
    point3 = np.linalg.norm(drivermouth[4] - drivermouth[3])
    Ypoint = (point1 + point2 + point3) / 3.0
    mouth_aspect_ratio = Ypoint / point
    return mouth_aspect_ratio

# Start the Video
video = cv2.VideoCapture(0)

# Setup Consecutive Frames Threshold
fps = video.get(cv2.CAP_PROP_FPS)
EAR_CONSEC_FRAMES = fps
MOR_CONSEC_FRAMES = fps

# Loop over the frames
while True:
    ret, frame = video.read()
    if not ret:
        break

    # Perform face detection
    result = face_detector.detect_faces(frame)

    # Loop over the detected faces
    for face in result:
        x, y, width, height = face['box']
        keypoints = face['keypoints']
        
        # Extract eye, mouth, and nose coordinates
        left_eye = np.array([keypoints['left_eye'][0], keypoints['left_eye'][1]])
        right_eye = np.array([keypoints['right_eye'][0], keypoints['right_eye'][1]])
        mouth = np.array([keypoints['mouth_left'][0], keypoints['mouth_left'][1]])
        nose = np.array([keypoints['nose'][0], keypoints['nose'][1]])
        
        # Draw bounding box and keypoints
        cv2.rectangle(frame, (x, y), (x + width, y + height), (255, 0, 255), 2)
        cv2.circle(frame, tuple(keypoints['left_eye']), 2, (0, 255, 0), 2)
        cv2.circle(frame, tuple(keypoints['right_eye']), 2, (0, 255, 0), 2)
        cv2.circle(frame, tuple(keypoints['nose']), 2, (0, 255, 0), 2)
        cv2.circle(frame, tuple(keypoints['mouth_left']), 2, (0, 255, 0), 2)

        # Calculate aspect ratios
        left_ear = EAR(np.array([left_eye, keypoints['left_eye'], keypoints['mouth_left'], keypoints['mouth_left']]))
        right_ear = EAR(np.array([right_eye, keypoints['right_eye'], keypoints['mouth_left'], keypoints['mouth_left']]))
        ear = (left_ear + right_ear) / 2.0
        mor = MOR(np.array([keypoints['mouth_left'], keypoints['mouth_right'], keypoints['mouth_left']]))
        
        # Logic for detecting drowsiness
        if ear < EAR_THRESHOLD:
            blink_consec_count += 1
        else:
            blink_consec_count = 0

        if mor > MOR_THRESHOLD:
            yawn_consec_count += 1
        else:
            yawn_consec_count = 0

        if blink_consec_count >= EAR_CONSEC_FRAMES or yawn_consec_count >= MOR_CONSEC_FRAMES:
            playsound(drowsy)
            engine.say("You are drowsy. I recommend you to take some rest or go for a walk.")
            engine.runAndWait()

    # Show the video frame
    cv2.imshow("Driver Drowsiness Monitoring System", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video.release() 
cv2.destroyAllWindows()
