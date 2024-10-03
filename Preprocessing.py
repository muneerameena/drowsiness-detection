# -*- coding: utf-8 -*-
"""
Created on Sat Aug 20 10:19:28 2022

@author: USER
"""
import cv2
import numpy as np
import math

def preprocessing(image):
    
    # Resize the Frames
    height = image.shape[0]
    width = image.shape[1]
    
    resize = bool()
    if height > 960:
        image_resize = cv2.resize(image, (int(960 * width/height), 960))
        resize = True
    elif width > 1920:
        image_resize = cv2.resize(image, (int(1920, 1920 * height/width)))
    else:
        resize = False
    
    
    # Perform Gamma Correction
    max_value = 255
    if resize == True:
        mean = np.mean(image_resize)
        gamma = math.log(mean)/math.log(max_value/2)
        
        image_gamma = np.array(((image_resize/255)**gamma)*255, dtype='uint8')
        
    else:
        mean = np.mean(image)
        gamma = math.log(mean)/math.log(max_value/2)
        
        image_gamma = np.array(((image/255)**gamma)*255, dtype='uint8')
        
    # Convert Color Image to Grayscale Image
    gray = cv2.cvtColor(image_gamma, cv2.COLOR_BGR2GRAY)
    
    # Plot the Preprocessing Steps
    '''cv2.imshow("Original Image", image)
    if resize == True:
        cv2.imshow("Resized Image", image_resize)
    cv2.imshow("After Gamma Correction", image_gamma)
    cv2.imshow("Grayscale Image", gray)'''
    
    return image_gamma, gray


# Example
img = cv2.imread("frame.jpg")

image_new, image_gray = preprocessing(img)

cv2.waitKey(0)