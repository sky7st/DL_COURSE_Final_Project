import cv2
import numpy as np
from statics import *
import mediapipe as mp

mp_hands_connections = mp.solutions.hands.HAND_CONNECTIONS

def drawData(data, image=None, drawConnect=True):
    if image is None:
        image = np.zeros((cam_h, cam_w, 3), dtype=np.float)
    if drawConnect:
        for connection in mp_hands_connections:
            start_idx = connection[0]
            end_idx = connection[1]

            start_x = int(data[start_idx][0] * cam_w)
            start_y = int(data[start_idx][1] * cam_h)

            end_x = int(data[end_idx][0] * cam_w)
            end_y = int(data[end_idx][1] * cam_h)
            cv2.line(image, 
                    (start_x, start_y), 
                    (end_x, end_y), 
                    color=(255, 0, 0), thickness=2)
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

def predictFilter(buffer):
    cnt = 0
    if len(buffer) < 10:
        return False
    else:
        for data in buffer:
            if data > 0.8:
                cnt += 1
        if cnt >= 10:
            return True
        else:
            return False
    
