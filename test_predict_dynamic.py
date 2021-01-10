import cv2, os, sys, time, h5py
import mediapipe as mp
import numpy as np
import tensorflow as tf
from utils import *

import matplotlib.pyplot as plt

labels = ['MouseLeft', 'MouseMove', 'MouseRight']

avg_cnt = 40

model = tf.keras.models.load_model('model_dynamic_mouse_hs.h5')

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# For webcam input:
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7, 
    min_tracking_confidence=0.7
)

# cap = cv2.VideoCapture(cv2.CAP_DSHOW)
cap = cv2.VideoCapture("test_dynamic.mkv")

cap_type = cap.get(cv2.CAP_PROP_BACKEND)

frame_cnt = 0
feature_cnt = 0
predict_cnt = 0
none_feature_cnt = 0


total_frame = []
result_ls = []
result_0, result_1, result_2 = [], [], []
conv_fifo_left, conv_fifo_right = [], []


first_draw = False

fig = plt.figure(figsize=(12, 6))
ax = plt.axes()
ax.set_ylim([-0.1, 1.1])
plt.legend()
plt.xlabel("Feature Frame Cnt")
plt.ylabel("Prediction")

while cap.isOpened():
    success, image = cap.read()
    if not success:
        if cap_type == cv2.CAP_DSHOW:
            print("Ignoring empty camera frame.")
            continue
        else:
            print("video end")
            break

    # Flip the image horizontally for a later selfie-view display, and convert
    # the BGR image to RGB.
    # if cap_type == cv2.CAP_DSHOW:
    image = cv2.flip(image, 1)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    results = hands.process(image)

    # Draw the hand annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    
    one_frame_feature = []
    if results.multi_hand_landmarks:
        # for hand_landmarks in results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        mp_drawing.draw_landmarks(image, hand_landmarks)
        
        for i, data_point in enumerate(hand_landmarks.landmark):
            one_frame_feature.append(
                np.array([data_point.x, data_point.y, data_point.z])
            )
        one_frame_feature = np.array(one_frame_feature)
        one_frame_feature_shift = calShift(one_frame_feature)
        drawData(one_frame_feature_shift, image)
        if len(total_frame) < avg_cnt:
            total_frame.append(one_frame_feature_shift)
        else:
            total_frame.pop(0)
            total_frame.append(one_frame_feature_shift)

        if len(total_frame) == avg_cnt:
            total_frame_np = np.array(total_frame)
            x_test = np.expand_dims(total_frame_np, axis=0)

            result = model.predict(x_test)
            result_ls.append(result[0])

            result_0.append(result[0][0])
            result_1.append(result[0][1])
            result_2.append(result[0][2])

            result_index = result[0].argmax()
            # if result[0][result_index] < 0.6:
            #     result_index = 1

            cv2.putText(image, "Result:{} {}".format(result_index, labels[result_index]), (10, 30), 1, 2, (0, 0, 255), 2)
            predict_cnt += 1

        feature_cnt += 1
        
    else:
        none_feature_cnt += 1
        if none_feature_cnt > 15:
            total_frame = []

    cv2.imshow('MediaPipe Hands', image)
    if cv2.waitKey(5) & 0xFF == 27:
        break

hands.close()
cap.release()


### draw plot
ax.plot(result_0, color='blue', label='left click') 
ax.plot(result_1, color='black', label='moving')
ax.plot(result_2, color='red', label='right click')

plt.show()
# plt.close(fig)