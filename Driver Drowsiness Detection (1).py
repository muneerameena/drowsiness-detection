# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 10:19:37 2022

@author: USER
"""

import os
from scipy.spatial import distance as dist
from imutils import face_utils
import numpy as np
import dlib
import cv2
import pyttsx3
from playsound import playsound
from Preprocessing import preprocessing



# Python text to speech parameters
engine = pyttsx3.init()
voices = engine.getProperty('voices')
engine.setProperty('rate', 150)
engine.setProperty('volume', 1)
engine.setProperty('voice', voices[1].id)

#Audio Files
alarm = audio_file = os.path.dirname(__file__) + "\\alarm.wav"
drowsy = audio_file = os.path.dirname(__file__) + "\\drowsy.mp3"


# Features
def EAR(drivereye):
    point1 = dist.euclidean(drivereye[1], drivereye[5])
    point2 = dist.euclidean(drivereye[2], drivereye[4])
    
    distance = dist.euclidean(drivereye[0], drivereye[3])
    
    ear_aspect_ratio = (point1 + point2) / (2.0 * distance)
    
    return ear_aspect_ratio


def MOR(drivermouth):
    
    point = dist.euclidean(drivermouth[0], drivermouth[6])
    
    point1 = dist.euclidean(drivermouth[2], drivermouth[10])
    
    point2 = dist.euclidean(drivermouth[3], drivermouth[9])
    
    point3 = dist.euclidean(drivermouth[4], drivermouth[8])

    Ypoint = (point1 + point2 + point3) / 3.0
    
    mouth_aspect_ratio = Ypoint / point
    
    return mouth_aspect_ratio


def NLR(drivernose, avg):
    point = dist.euclidean(drivernose[0], drivernose[3])
    
    
    nose_length_ratio = point / avg
    
    return nose_length_ratio


# Setup the Threshold
EAR_THRESHOLD = 0.24

MOR_THRESHOLD = 0.6

NLR_LOWER_THRESHOLD = 0.75
NLR_UPPER_THRESHOLD = 1.3

MOE_THRESHOLD = 2.4


# Setup the audio warning limit
WARN_LIMIT = 5


# Initialize the parameters
blink_consec_count = 0
yawn_consec_count = 0

yawn_count = 0
yawn_count1 = 0

drowsy_blink_count = 0
drowsy_blink_count1 = 0


# Initialize dlib's Face Detector and Facial Landmark Predictor
shape_predictor_path = "shape_predictor_68_face_landmarks.dat"

detector = dlib.get_frontal_face_detector()

predictor = dlib.shape_predictor(shape_predictor_path)


# Provide indexes for the required Facial Landmarks
(nStart, nEnd) = (27, 31)
# (nStart, nEnd) = face_utils.FACIAL_LANDMARKS_IDXS['nose']
# nEnd = nEnd - 5

(rStart, rEnd) = (36, 42)
# (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

(lStart, lEnd) = (42, 48)
# (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]

(mStart, mEnd) = (48, 68)
# (mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]


# Start the Video
video = cv2.VideoCapture(0)


# Setup Consecutive Frames Threshold
fps = video.get(cv2.CAP_PROP_FPS)

EAR_CONSEC_FRAMES = fps
EAR_CONSEC_FRAMES_ALERT = fps * 5

MOR_CONSEC_FRAMES = fps


# Loop over the frames
while True:
    ret, frame = video.read()
    
    # Perform preprocessing
    frame, gray = preprocessing(frame)
    
    # Detect faces in image
    rects = detector(gray, 0)


    # Loop over the detected faces
    for rect in rects:
        
        # Extract the Face Coordinates and Visualize
        face_coordinates = np.array([[rect.tl_corner().x, rect.tl_corner().y],
                             [rect.tr_corner().x, rect.tr_corner().y],
                             [rect.bl_corner().x, rect.bl_corner().y],
                             [rect.br_corner().x, rect.br_corner().y]])
        
        
        face_coordinatesHull = cv2.convexHull(face_coordinates)
        cv2.drawContours(frame, [face_coordinatesHull], -1, (255, 0, 255), 1)
        
        
        # Determine Facial Landmarks and convert into numpy array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        
        
        # Extract the coordinates and Visualize them
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        mouth = shape[mStart:mEnd]
        nose = shape[nStart:nEnd]
        
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        mouthHull = cv2.convexHull(mouth)
        noseHull = cv2.convexHull(nose)
        
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [mouthHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [noseHull], -1, (0, 255, 0), 1)
        
        
        # Extract the features from the face
        left_ear = EAR(leftEye)
        right_ear = EAR(rightEye)
        ear = (left_ear + right_ear)/ 2.0
        
        mor = MOR(mouth)
        
        nlr = NLR(nose, avg=25)
        
        moe = mor/ear
        
        
        # Check Head Bending
        if nlr < NLR_LOWER_THRESHOLD or nlr > NLR_UPPER_THRESHOLD:
            
            
            # Check whether Yawning is also present
            if mor > MOR_THRESHOLD and moe > MOE_THRESHOLD:
                
                yawn_consec_count += 1
                
                if yawn_consec_count >= MOR_CONSEC_FRAMES:
                    
                    cv2.putText(frame, "Yawning and Head Bending Detected", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                    
                    cv2.putText(frame, "DROWSINESS DETECTED!", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
                else:
                    cv2.putText(frame, "Head Bending Detected", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                    
                    
            # Increment Yawn count if Yawning detected
            elif yawn_consec_count >= MOR_CONSEC_FRAMES:
                yawn_count += 1
                yawn_count1 += 1
        
                yawn_consec_count = 0
                
                cv2.putText(frame, "Head Bending Detected", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            
            
            # Check whether Drowsy Blinking is also present
            elif ear < EAR_THRESHOLD:
                
                blink_consec_count += 1
                
                
                cv2.putText(frame, "Eyes Closed ", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                if blink_consec_count >= EAR_CONSEC_FRAMES:
                    
                    cv2.putText(frame, "Drowsy Blinking and Head Bending Detected", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                    
                    cv2.putText(frame, "DROWSINESS DETECTED!", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
                    # Play Emergency Alarm if eyes closed for longer period
                    if blink_consec_count >= EAR_CONSEC_FRAMES_ALERT:
                        playsound(alarm)
                else:
                    cv2.putText(frame, "Head Bending Detected", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                   
                    
            # Increment Drowsy Blink Count if Drowsy Blink detected       
            elif blink_consec_count >= EAR_CONSEC_FRAMES:
                drowsy_blink_count += 1
                drowsy_blink_count1 += 1
                
                blink_consec_count = 0
                
                cv2.putText(frame, "Head Bending Detected", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                
            else:
                cv2.putText(frame, "Head Bending Detected", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                
        
        # Check whether Yawning is present
        elif mor > MOR_THRESHOLD and moe > MOE_THRESHOLD:
            
            yawn_consec_count += 1
            
            if yawn_consec_count >= MOR_CONSEC_FRAMES:
                
                cv2.putText(frame, "Yawning Detected", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                
                cv2.putText(frame, "DROWSINESS DETECTED!", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
            
        # Increment Yawn count if Yawning detected
        elif yawn_consec_count >= MOR_CONSEC_FRAMES:
            yawn_count += 1
            yawn_count1 += 1
    
            yawn_consec_count = 0
        
        
        # Check whether Drowsy Blinking is present
        elif ear < EAR_THRESHOLD:
            
            blink_consec_count += 1
            
            cv2.putText(frame, "Eyes Closed ", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            if blink_consec_count >= EAR_CONSEC_FRAMES:
                
                cv2.putText(frame, "Drowsy Blinking Detected", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                
                cv2.putText(frame, "DROWSINESS DETECTED!", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
                # Play Emergency Alarm if eyes closed for longer period
                if blink_consec_count >= EAR_CONSEC_FRAMES_ALERT:
                    playsound(alarm)
               
                
        # Increment Drowsy Blink Count if Drowsy Blink detected       
        elif blink_consec_count >= EAR_CONSEC_FRAMES:
            drowsy_blink_count += 1
            drowsy_blink_count1 += 1
            
            blink_consec_count = 0
                

        # Visualize values of features
        cv2.putText(frame, "EAR: {:.2f}".format(ear), (480, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        cv2.putText(frame, "MOR: {:.2f}".format(mor), (480, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        cv2.putText(frame, "NLR: {:.2f}".format(nlr), (480, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        cv2.putText(frame, "MOE: {:.2f}".format(moe), (480, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        
        # Visualize Total Number of Blinks and Total Number of Yawns
        cv2.putText(frame, "Blinks: {}".format(drowsy_blink_count1), (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        cv2.putText(frame, "Yawns: {}".format(yawn_count1), (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        
        # Play Warning Message
        if drowsy_blink_count + yawn_count > 5:
            print("playing sound")
            playsound(drowsy)
            engine.say("You are drowsy. I recommend you to take some rest or go for a walk.")
            engine.runAndWait()
            
            drowsy_blink_count = 0
            yawn_count = 0
            
    
    # Visualize Frame
    cv2.imshow("Driver Drowsiness Monitoring System", frame)
    
    
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video.release() 
cv2.destroyAllWindows()