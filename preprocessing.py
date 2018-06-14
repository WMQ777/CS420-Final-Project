'''
Preprocessing by moving noise and centralization
'''
#-*-coding:utf-8-*-
import os
import numpy as np
from skimage import io,data,measure,color,filters,morphology,segmentation
from PIL import Image
import matplotlib.pyplot as plt


data_num = 10000
fig_w = 45

data = np.fromfile( "mnist_test_data0",dtype=np.uint8)
label = np.fromfile( "mnist_test_label",dtype=np.uint8)

print(data.shape)
print(label.shape)

data = data.reshape(data_num,fig_w,fig_w)

print("After reshape:",data.shape)
imagenum = data.shape[0]

resfile = np.zeros([data_num,fig_w,fig_w], dtype=np.uint8)
for i in range(imagenum):
    print(i)
    img = data[i]
    m, n = img.shape

    bw = np.zeros([m, n])
    for k in range(m):
        for j in range(n):
            if img[k][j] > 0:
                bw[k][j] = 1

    labels = measure.label(bw, connectivity=2)
    info = measure.regionprops(labels)
    num = labels.max()
    area = []
    for k in range(num):
        area.append(info[k].area)
    maxindex = area.index(max(area))

    bbox = info[maxindex].bbox
    x1 = bbox[1]
    y1 = bbox[0]
    x2 = bbox[3]
    y2 = bbox[2]
    mask = (labels == (maxindex + 1))
    tmp = mask * img
    number = tmp[y1: y2, x1: x2]

    w, h = number.shape
    result = np.zeros([m, n], dtype=np.uint8)

    startr = int((45 - w) / 2)
    endr = startr + w
    startc = int((45 - h) / 2)
    endc = startc + h

    result[startr:endr, startc:endc] = number
    data[i] = result

data.tofile("mnist_test_data")