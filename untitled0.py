# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 08:09:46 2023

@author: USER
"""

import mtcnn
from mtcnn.mtcnn import MTCNN
import cv2

def detect_face(image):
    detector = MTCNN()
    bounding_boxes = detector.detect_faces(image)
    return bounding_boxes
def draw_bb(image, bboxes):
    for box in bboxes:
        x1, y1, w, h = box['box']
        cv2.rectangle(image, (x1, y1), (x1+w, y1+h), (0, 255, 0), 2)

detector = MTCNN()
video = cv2.VideoCapture(1)
while True:
    ret, frame = video.read()
    bboxes = detect_face(frame)
    for box in bboxes:
        x1, y1, w, h = box['box']
        cv2.rectangle(frame, (x1, y1), (x1+w, y1+h), (0, 255, 0), 2)
        roi = frame[y1:y1+h, x1:x1+w]
    final = cv2.resize(roi, (48, 48, 3))
    cv2.imshow("Driver Drowsiness Monitoring System", frame)
    
    
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video.release() 
cv2.destroyAllWindows()
    