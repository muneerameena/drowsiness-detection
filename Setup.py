# -*- coding: utf-8 -*-
"""
Created on Tue Aug 23 17:16:40 2022

@author: USER
"""

from scipy.spatial import distance as dist
from imutils import face_utils
import numpy as np
import dlib
import cv2
import pandas as pd
pd.options.mode.chained_assignment = None
import winsound
from Preprocessing import preprocessing
import matplotlib.pyplot as plt
import statistics

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


# Initialize dlib's Face Detector and Facial Landmark Predictor
shape_predictor_path = "shape_predictor_68_face_landmarks.dat"

detector = dlib.get_frontal_face_detector()

predictor = dlib.shape_predictor(shape_predictor_path)


current_frame = 0
frame_info_list = []
frame_counter = 0


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
 
fps = video.get(cv2.CAP_PROP_FPS)

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
        
        nlr = NLR(nose, avg=1)
        
        moe = mor/ear
        
        frame_info = {
            'frame_no': current_frame,
            'EAR': ear,
            'MOR': mor,
            'NLR': nlr,
            'MOE': moe,
            }
        
        # Visualize values of features
        cv2.putText(frame, "EAR: {:.2f}".format(ear), (480, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        cv2.putText(frame, "MOR: {:.2f}".format(mor), (480, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        cv2.putText(frame, "NLR: {:.2f}".format(nlr), (480, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        cv2.putText(frame, "MOE: {:.2f}".format(moe), (480, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        
        # Categorize each 20 seconds to each state
        if frame_counter < 600:
            cv2.putText(frame, "Normal", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        elif frame_counter < 1200:
            cv2.putText(frame, "Drowsy Blinking", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        elif frame_counter < 1800:
            cv2.putText(frame, "Yawning", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        elif frame_counter < 2400:
            cv2.putText(frame, "Head Bending Forward", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        elif frame_counter < 3000:
            cv2.putText(frame, "Head Bending Backward", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        
        # Play a beep sound in state conversion
        frame_array = [600, 1200, 1800, 2400]
        
        if frame_counter in frame_array:
            winsound.Beep(1000, 50)
            
        frame_counter+= 1
        
        # Append processed frames to frame info list
        frame_info_list.append(frame_info)
        
        # Show the frame
        cv2.imshow("Frame", frame)
        
        
    # if 'q' key was pressed, break from the loop    
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
    
    current_frame += 1
    
    # Build a dataframe from frame information list
    frame_info_df = pd.DataFrame(frame_info_list)
    
    
    # Stopping Criteria
    up_to = 3000
    if up_to == current_frame:
        break
        

# a bit of clean-up
cv2.destroyAllWindows()
video.release()


# Classify dataframe to each state
normal = frame_info_df.iloc[:600, :]
normal.reset_index(drop=True, inplace=True)
normal.rename(columns = {'EAR':'EAR-normal', 'MOR': 'MOR-normal', 'MOE': 'MOE-normal', 'NLR': 'NLR-normal'}, inplace=True)

blink = frame_info_df.iloc[600:1200, :]
blink.reset_index(drop=True, inplace=True)
blink.rename(columns = {'EAR' : 'EAR-drowsy'}, inplace=True)

yawn = frame_info_df.iloc[1200:1800, :]
yawn.reset_index(drop=True, inplace=True)
yawn.rename(columns = {'MOR': 'MOR-drowsy', 'MOE': 'MOE-drowsy'}, inplace=True)

bend_forward = frame_info_df.iloc[1800:2400, :]
bend_forward.reset_index(drop=True, inplace=True)
bend_forward.rename(columns = {'NLR' : 'NLR-forward'}, inplace=True)

bend_backward = frame_info_df.iloc[2400:3000, :]
bend_backward.reset_index(drop=True, inplace=True)
bend_backward.rename(columns = {'NLR' : 'NLR-backward'}, inplace= True)


# Calculate Average nose length
nlr_mean = statistics.mean(normal["NLR-normal"].values.tolist())
print("Average Nose Length : ", nlr_mean)


# Join each features together to form a dataframe
eye = pd.concat([normal["EAR-normal"], blink["EAR-drowsy"]], axis = 1)
mouth = pd.concat([normal["MOR-normal"], yawn["MOR-drowsy"]], axis = 1)
mouth1 = pd.concat([normal["MOE-normal"], yawn["MOE-drowsy"]], axis = 1)
bend = pd.concat([normal["NLR-normal"], bend_forward["NLR-forward"], bend_backward["NLR-backward"]], axis = 1)


# Divide NLR with Average nose length
bend["NLR-normal"] = bend["NLR-normal"] / nlr_mean
bend["NLR-forward"] = bend["NLR-forward"] / nlr_mean
bend["NLR-backward"] = bend["NLR-backward"] / nlr_mean


# Plot each features
eye.loc[:,['EAR-normal','EAR-drowsy']].plot(secondary_y=['B'], mark_right=False, figsize = (15,10), grid=True)
plt.title("Eye Aspect Ratio (EAR)")
plt.ylabel("EAR")
plt.xlabel("Frames")
plt.show()

mouth.loc[:,['MOR-normal','MOR-drowsy']].plot(secondary_y=['B'], mark_right=False, figsize = (15,10), grid=True)
plt.title("Mouth Opening Ratio (MOR)")
plt.ylabel("MOR")
plt.xlabel("Frames")
plt.show()

mouth1.loc[:,['MOE-normal','MOE-drowsy']].plot(secondary_y=['B'], mark_right=False, figsize = (15,10), grid=True)
plt.title("Mouth over Eye Ratio (MOE)")
plt.ylabel("MOE")
plt.xlabel("Frames")
plt.show()

bend.loc[:,['NLR-normal','NLR-forward', 'NLR-backward']].plot(secondary_y=['B'], mark_right=False, figsize = (15,10), grid=True)
plt.title("Nose Length Ratio (NLR)")
plt.ylabel("NLR")
plt.xlabel("Frames")
plt.show()
