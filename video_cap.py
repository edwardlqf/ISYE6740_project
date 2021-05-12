# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 14:37:31 2020

@author: Edward
"""

import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    
    
    cv2.imshow('frame', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
        
cap.release()
cv2.destroyAllWindows