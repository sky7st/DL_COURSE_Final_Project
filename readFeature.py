import cv2, os, sys, time, h5py
import mediapipe as mp
import numpy as np

label_basics = ["VirtualMouse", "VirtualSelect", "Other"]

filename = "data.h5"

labels = []
show_data_index = 0
show_key_index = 0

cam_h = 480
cam_w = 640

def drawData(data):
    empty_img = np.zeros((cam_h, cam_w, 3), dtype=np.float)
    for i in range(data.shape[0]):
        x = int(data[i][0] * cam_w)
        y = int(data[i][1] * cam_h)
        empty_img = cv2.circle(empty_img, (x, y), radius=0, color=(0, 0, 255), thickness=4)
    return empty_img
with h5py.File(filename, 'r') as h5f:
    
    labels = list(h5f.keys())
    max_label_index = len(labels) - 1

    label = labels[show_key_index]
    dataset = h5f[label][()]

    max_data_index = dataset.shape[0] - 1

    data = dataset[show_data_index]
    img = drawData(data)
    cv2.putText(img, "index: {}".format(show_data_index), (10, 30), 0, 1, (255, 0, 0))
    cv2.putText(img, "label: {}".format(label), (10, 55), 0, 1, (255, 0, 0))
    cv2.putText(img, "shape:{}".format(data.shape), (10, 80), 0, 1, (255, 0, 0))
    cv2.imshow('MediaPipe Hands', img)
    while True:
        try:
            key =  cv2.waitKeyEx(5)
            if key == 2490368 or key == 2621440:
                if key == 2490368: #ARROW_UP
                    if show_key_index >= max_label_index:
                        show_key_index = 0
                    else:
                        show_key_index += 1
                elif key == 2621440: #ARROW_DOWN
                    if show_key_index <= 0:
                        show_key_index = max_label_index
                    else:
                        show_key_index -= 1
                label = labels[show_key_index]
                dataset = h5f[label][()]
                max_data_index = dataset.shape[0] - 1
                show_data_index = 0
                data = dataset[show_data_index]
                img = drawData(data)
                cv2.putText(img, "index: {}".format(show_data_index), (10, 30), 0, 1, (255, 0, 0))
                cv2.putText(img, "label: {}".format(label), (10, 55), 0, 1, (255, 0, 0))
                cv2.putText(img, "shape:{}".format(data.shape), (10, 80), 0, 1, (255, 0, 0))
                cv2.imshow('MediaPipe Hands', img)
            elif key == 2424832 or key == 2555904:
                if key == 2424832: #ARROW_LEFT
                    if show_data_index <= 0:
                        show_data_index = max_data_index
                    else:
                        show_data_index -= 1
                elif key == 2555904: #ARROW_RIGHT
                    if show_data_index >= max_data_index:
                        show_data_index = 0
                    else:
                        show_data_index += 1
                data = dataset[show_data_index]
                img = drawData(data)
                cv2.putText(img, "index: {}".format(show_data_index), (10, 30), 0, 1, (255, 0, 0))
                cv2.putText(img, "label: {}".format(label), (10, 55), 0, 1, (255, 0, 0))
                cv2.putText(img, "shape:{}".format(data.shape), (10, 80), 0, 1, (255, 0, 0))
                cv2.imshow('MediaPipe Hands', img)
            else:
                if key == 27:
                    break
        except KeyboardInterrupt:
            break

