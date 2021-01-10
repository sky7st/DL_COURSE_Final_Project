import cv2
import numpy as np
from statics import *
import mediapipe as mp
import win32api, time

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
    
def handInROI(data, img, roi=(170, 100, 300, 280)):
    shape = img.shape
    min_arr = np.argmin(data, axis=0)
    max_arr = np.argmax(data, axis=0)
    min_x = int(data[min_arr[0],0] * shape[1]) ##X
    min_y = int(data[min_arr[1],1] * shape[0]) ##Y

    max_x = int(data[max_arr[0],0] * shape[1]) ##X
    max_y = int(data[max_arr[1],1] * shape[0]) ##Y

    if (min_x >= roi[0] and min_y >= roi[1]) and (max_x <= roi[0] + roi[2] and max_y <= roi[1] + roi[3]):
        return True
    else:
        return False

def getCoord(data, shape=(640, 480)):
    data_x = int(data[0] * shape[0])
    data_y = int(data[1] * shape[1])
    return (data_x, data_y)

def moveFromTo(p1, p2):
    if len(win32api.EnumDisplayMonitors(None, None)) > 1:
        (hMon, hDC, (left, top, right, bottom)) = win32api.EnumDisplayMonitors(None, None)[1]
        if left < 0:
            p1 = (p1[0]+left, p1[1])
            p2 = (p2[0]+left, p2[1])
    try:
        # slope of our line
        m = (p2[1] - p1[1]) / (p2[0] - p1[0])
        # y intercept of our line
        i = p1[1] - m * p1[0]
        # current point
        cP = list(p1)
        # while loop comparison
        comp = isGreater
        # moving left to right or right to left
        inc = -1
        # switch for moving to right
        if (p2[0] > p1[0]):
            comp = isLess
            inc = 1
        # move cursor one pixel at a time
        while comp(cP[0],p2[0]):
            win32api.SetCursorPos(cP)
            cP[0] += inc
            # get next point on line
            cP[1] = m * cP[0] + i
            # slow it down
            time.sleep(0.01)
    except:
        pass

def isLess(a,b):
    return a < b
def isGreater(a,b):
    return a > b
