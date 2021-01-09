import cv2, os, sys, time, h5py
import mediapipe as mp
import numpy as np
from utils import *
filename = "data_test.h5"

labels = []
show_data_index = 0
show_key_index = 0


with h5py.File(filename, 'r+') as h5f:
    
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
                if key == 27: ##ESC
                    break
                elif key == 100: ##d  ##delete one index data
                    print("D")
                    dataset = h5f[label][()]
                    
                    res = np.delete(dataset, show_data_index, axis=0)  # delete element with index 1, i.e. second element
                    
                    h5f.__delitem__(label)  # delete existing dataset
                    if res.size != 0:
                        h5f[label] = res  # reassign to dataset
                        dataset = h5f[label][()]
                        max_data_index = dataset.shape[0] - 1
                        if show_data_index >= max_data_index:
                            show_data_index = 0
                        else:
                            show_data_index += 1
                    else:
                        labels = list(h5f.keys())
                        max_label_index = len(labels) - 1
                        if show_key_index >= max_label_index:
                            show_key_index = 0
                        else:
                            show_key_index += 1
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

                elif key == 102: ##f delete this label
                    del h5f[label]
                    labels = list(h5f.keys())
                    max_label_index = len(labels) - 1
                    if show_key_index >= max_label_index:
                        show_key_index = 0
                    else:
                        show_key_index += 1
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
        except KeyboardInterrupt:
            break

