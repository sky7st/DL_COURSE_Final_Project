import cv2
import numpy as np
from statics import *

def drawData(data, image=None):
    if image is None:
        image = np.zeros((cam_h, cam_w, 3), dtype=np.float)
    for i in range(data.shape[0]):
        x = int(data[i][0] * cam_w)
        y = int(data[i][1] * cam_h)
        image = cv2.circle(image, (x, y), radius=0, color=(0, 0, 255), thickness=4)
    return image

def calShift(data):
    data_copy = data.copy()
    shift_arr = np.argmin(data, axis=0)
    data_copy[:,0] -= data[shift_arr[0],0] ##X
    data_copy[:,1] -= data[shift_arr[1],1] ##Y
    data_copy[:,2] -= data[shift_arr[2],2] ##Z
    return data_copy