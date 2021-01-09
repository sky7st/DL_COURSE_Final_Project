import cv2, os, sys, time, h5py
import mediapipe as mp
import numpy as np
from utils import *
filename = "data_dynamic.h5"

fps = 30.0
timeout_ms = int(1000/fps)

labels = []
show_data_index = 0
show_key_index = 0
end_flag = False
with h5py.File(filename, 'r+') as h5f:
    
    labels = list(h5f.keys())

    max_label_index = len(labels) - 1

    label = labels[show_key_index]
    datasets = h5f[label]

    datasets_keys = list(datasets.keys())
    max_data_index = len(datasets_keys) - 1
    datas = datasets[datasets_keys[show_data_index]]

    for i in range(datas.shape[0]): #frame cnt
        data = datas[i]
        img = drawData(data)
        cv2.putText(img, "index: {}, frame_cnt: {}".format(show_data_index, i), (10, 30), 0, 1, (255, 0, 0))
        cv2.putText(img, "label: {}, file: {}".format(label, datasets_keys[show_data_index]), (10, 55), 0, 0.8, (255, 0, 0))
        cv2.putText(img, "shape:{}".format(datas.shape), (10, 80), 0, 1, (255, 0, 0))
        cv2.imshow('MediaPipe Hands', img)

        if cv2.waitKeyEx(timeout_ms) & 0xFF == 27:
            end_flag = True
            break
    if not end_flag:
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
                    datasets = h5f[label]

                    datasets_keys = list(datasets.keys())
                    max_data_index = len(datasets_keys) - 1
                    show_data_index = 0
                    datas = datasets[datasets_keys[show_data_index]]

                    for i in range(datas.shape[0]): #frame cnt
                        data = datas[i]
                        img = drawData(data)
                        cv2.putText(img, "index: {}, frame_cnt: {}".format(show_data_index, i), (10, 30), 0, 1, (255, 0, 0))
                        cv2.putText(img, "label: {}, file: {}".format(label, datasets_keys[show_data_index]), (10, 55), 0, 0.8, (255, 0, 0))
                        cv2.putText(img, "shape:{}".format(datas.shape), (10, 80), 0, 1, (255, 0, 0))
                        cv2.imshow('MediaPipe Hands', img)

                        if cv2.waitKeyEx(timeout_ms):
                            pass
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
                    datas = datasets[datasets_keys[show_data_index]]

                    for i in range(datas.shape[0]): #frame cnt
                        data = datas[i]
                        img = drawData(data)
                        cv2.putText(img, "index: {}, frame_cnt: {}".format(show_data_index, i), (10, 30), 0, 1, (255, 0, 0))
                        cv2.putText(img, "label: {}, file: {}".format(label, datasets_keys[show_data_index]), (10, 55), 0, 0.8, (255, 0, 0))
                        cv2.putText(img, "shape:{}".format(datas.shape), (10, 80), 0, 1, (255, 0, 0))
                        cv2.imshow('MediaPipe Hands', img)

                        if cv2.waitKeyEx(timeout_ms):
                            pass
                else:
                    if key == 27: ##ESC
                        break
            except KeyboardInterrupt:
                break

