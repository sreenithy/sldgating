'''
If size of a projection is pxq
With one pixel turned on at a time capture pq number of images
Suppose camera resolution is mxn
Reshape it to (mn,1) and stack all the images columnwise corresponding to the number of illumination patterns
This gives light transport matrix T of size (mnxpq)

'''
import imageio

import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import glob
import re
import os
from functions.exr2png import *
numbers = re.compile(r'(\d+)')

def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts


def relight_static_grayscale(xres, proj_width, output):
    [w, h] = [xres, xres]
    T = []
    cnt = 0
    for filename in sorted(glob.glob(output + "/scene_*.jpg"), key=numericalSort):
        img = cv2.imread(filename, 0)
        [w, h] = np.shape(img)
        img = np.reshape(img, (w * h, 1))
        T.append(img)
        cnt = cnt + 1
    T = np.transpose(T)
    T = np.reshape(T, (w * h, cnt))
    mask = np.ones((proj_width,proj_width))
    mask = np.reshape(mask, (proj_width * proj_width, 1))
    c = np.matmul(T, mask)
    c = c.reshape((xres, xres))


    norm = (c - np.min(c)) / (np.max(c) - np.min(c))
    cv2.imwrite(output + "/lt_reconstruction.png", norm*255)

    #Dual
    Ttranspose = np.transpose(T)
    c__ = np.ones((w*h,1))
    p__ = np.matmul(Ttranspose,c__)
    p__ = p__.reshape((proj_width,proj_width))
    p__ = (p__ - np.min(p__)) / (np.max(p__) - np.min(p__))
    cv2.imwrite(output + "/lt_dual.png", p__*255)

def processT(T1, w, h, cnt, mask, xres):
    T1 = np.transpose(T1)
    T1 = np.reshape(T1, (w * h, cnt))
    c1 = np.matmul(T1, mask) * 1
    c1 = c1.reshape((xres, xres))
    return c1

def relight_static_rgb(xres, proj_width, output):
    '''
    Given dot-scans obtain the full light-transport
    :param xres: image resolution
    :param proj_width: projector resolution
    :param output: path to folder with dot scans
    :return:
    '''

    [w, h] = [xres, xres]
    T1 = []
    T2 = []
    T3 = []
    cnt = 0
    c = np.zeros((xres,xres,3))

    for filename in sorted(glob.glob(output + "/scene_*.exr"), key=numericalSort):
        img = cv2.imread(filename,-1)
        [w, h,_] = np.shape(img)
        img1 = np.reshape(img[:, :, 0], (w * h, 1))
        img2 = np.reshape(img[:, :, 1], (w * h, 1))
        img3 = np.reshape(img[:, :, 2], (w * h, 1))
        T1.append(img1)
        T2.append(img2)
        T3.append(img3)
        cnt = cnt + 1

    mask = np.ones((proj_width,proj_width))
    mask = np.reshape(mask, (proj_width * proj_width, 1))

    c1 = processT(T1, w, h, cnt, mask, xres)
    c2 = processT(T2, w, h, cnt, mask, xres)
    c3 = processT(T3, w, h, cnt, mask, xres)

    c[:,:,0] = c3
    c[:,:,1] = c2
    c[:,:,2] = c1
    norm = (c - np.min(c)) / (np.max(c) - np.min(c))
    imageio.imwrite(output + "/lt_primal_rgb.exr", norm.astype('float32'))
    convert_exr2png(output + "/lt_primal_rgb.exr",output + "/lt_primal_rgb.png",'gamma')
    np.save(output +"/Tr.npy", T1 )
    np.save(output +"/Tg.npy", T2 )
    np.save(output +"/Tb.npy", T3 )

    #Dual
    p_ = np.zeros((proj_width,proj_width,3))
    Ttranspose = np.transpose(T1)
    c__ = np.ones((w*h,1))
    p__ = np.matmul(Ttranspose,c__)
    p__ = p__.reshape((proj_width,proj_width))
    p_[:,:,2] = p__

    Ttranspose = np.transpose(T2)
    c__ = np.ones((w*h,1))
    p__ = np.matmul(Ttranspose,c__)
    p__ = p__.reshape((proj_width,proj_width))
    p_[:,:,1] = p__

    Ttranspose = np.transpose(T3)
    c__ = np.ones((w*h,1))
    p__ = np.matmul(Ttranspose,c__)
    p__ = p__.reshape((proj_width,proj_width))
    p_[:,:,0] = p__
    p_ = (p_ - np.min(p_)) / (np.max(p_) - np.min(p_))
    imageio.imwrite(output + "/lt_dual_rgb.exr", p_.astype('float32'))
    convert_exr2png(output + "/lt_dual_rgb.exr",output + "/lt_dual_rgb.png")

