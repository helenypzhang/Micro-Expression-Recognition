#!/usr/bin/env python
# coding: utf-8

# In[4]:


from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import cv2
import matplotlib.pyplot as plt

# In[5]:


def face_align(image, rows, columns):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat\shape_predictor_68_face_landmarks.dat")
    fa = imutils.face_utils.FaceAligner(predictor, desiredFaceWidth = rows, desiredFaceHeight = columns)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)
    
    if len(rects) != 1:
        print(len(rects), " faces detected")
        # return None
    
    # extract the ROI of the *original* face, then align the face
    # using facial landmarks
    (x, y, w, h) = face_utils.rect_to_bb(rects[0])
    try:
        faceOrig = imutils.resize(image[y:y + h, x:x + w], width=rows, height = columns)
        faceAligned = fa.align(image, gray, rects[0])
        print("face aligned type: {}".format(type(face_align)))
        return faceAligned
    except Exception as e:
        print(str(e))
        print(e.with_traceback)
        return None

img = cv2.imread('img.jpg')
temp = face_align(img, 960, 640)
# plt.imsave('test2.png', temp)
print(temp)
