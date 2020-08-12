#!/usr/bin/env python2
import time
import argparse
import cv2
import itertools
import os

import numpy as np
from scipy import misc

import openface
import redis
import pickle
from PIL import Image


class load_lib(object):
    def __init__(self, face_cascade, network_model, dlib_face_predictor, dim=96):
        self.face_cascade = cv2.CascadeClassifier(face_cascade)
        self.net = openface.TorchNeuralNet(network_model, dim)
        self.align = openface.AlignDlib(dlib_face_predictor)
        pool = redis.ConnectionPool(host='localhost', port=6379, db=0)
        self.r = redis.Redis(connection_pool=pool)
        self.dim = dim

    def face_detect(self, img):
        faces = self.face_cascade.detectMultiScale(img, 1.3, 5)
        list_face = []
        for (x,y,w,h) in faces:
            sub_face = img[y:y+h, x:x+w]
            if sub_face.shape[0] > self.dim and sub_face.shape[1] > self.dim:
                cropped = misc.imresize(sub_face, (self.dim, self.dim), interp='bilinear')
                list_face.append(cropped)
        return list_face

    def align_function(self,img):
        # aligned_face = self.align.align(self.dim, img,landmarkIndices=openface.AlignDlib.INNER_EYES_AND_BOTTOM_LIP)
        aligned_face = self.align.align(self.dim, img,landmarkIndices=openface.AlignDlib.INNER_EYES_AND_BOTTOM_LIP)
        return aligned_face

    def get_vector(self,img):
        vector = self.net.forward(img)
        return vector

    def save_redis(self,img,vector):
        self.r.set(img,pickle.dumps(vector))


if __name__ == '__main__':
    lib = load_lib(face_cascade='/home/dsvn/workspace/openface/models/dlib/haarcascade_frontalface_default.xml'
        , networkModel='/home/dsvn/workspace/openface/models/openface/nn4.small2.v1.t7'
        , dlibFacePredictor='/home/dsvn/workspace/openface/models/dlib/shape_predictor_68_face_landmarks.dat')
    
    #########
    img_path = '/home/dsvn/Documents/hongnt/fb_image/HA_Phuong_Thao_files/476_221171921357240_1689212226_n.jpg'
    im = np.asarray(Image.open(img_path))
    face = lib.face_detect(im)
    for i in range(len(face)):
        misc.imsave("test_"+str(i)+".png", face[i])
        face_align = lib.align_function(np.array(face[i]))
        if face_align is not None:
            vector = lib.get_vector(face_align)
            lib.save_redis(img_path+"."+str(i) ,vector)

