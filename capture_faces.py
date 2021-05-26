#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 23:30:56 2019

@author: srieteja
"""
import cv2

cam = cv2.VideoCapture(0)
detector = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_default.xml')
Id = input('enter your id: ')
sampleNum = 0
while(True):
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        sampleNum = sampleNum+1
        cv2.imwrite(Id+'.'+str(sampleNum)+".jpg",cv2.resize
        (gray[y:y+h,x:x+w],(70,70)))
        cv2.imshow('frame',img)
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break
    elif sampleNum >= 100:
        break
cam.release()
cv2.destroyAllWindows()