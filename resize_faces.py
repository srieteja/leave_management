#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 15:22:57 2019

@author: srieteja
"""
import cv2,os
import numpy as np

base_dir = os.path.dirname(os.path.abspath("__file__"))
image_dir = os.path
recognizer = cv2.face.EigenFaceRecognizer_create()
detector= cv2.CascadeClassifier("haarcascade_frontalface_default.xml")


def getImagesAndLabels(path):
    width_d, height_d = 280, 280
    imagePaths=[os.path.join(path,f) for f in os.listdir(path)]
    faceSamples=[]
    ids=[]

    for imagePath in imagePaths:
        pilImage = Image.open(imagePath).convert('L')
        imageNp = np.array(pilImage,'uint8')
        Id = int(os.path.split(imagePath)[-1].split(".")[1])
        faces = detector.detectMultiScale(imageNp)
        for (x,y,w,h) in faces:
            faceSamples.append(cv2.resize(imageNp[y:y+h,x:x+w], (width_d, height_d))
            ids.append(Id)
    return faceSamples,ids
faces,ids = getImagesAndLabels('dataSet')
recognizer.train(faces, np.array(ids))
recognizer.write('trainner/trainnerEi.yml')