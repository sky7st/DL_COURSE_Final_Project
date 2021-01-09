import cv2, os, sys, time, h5py
import mediapipe as mp
import numpy as np
import tensorflow as tf
from utils import *

labels = ['VirtualFive', 'VirtualFour', 'VirtualGood', 
          'VirtualOK', 'VirtualOne', 'VirtualRotate', 
          'VirtualStone', 'VirtualThree', 'VirtualTwo']
# model = tf.keras.models.load_model('model.h5')
# model = tf.keras.models.load_model('model_4.h5')
model = tf.keras.models.load_model('model_9.h5')

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# For webcam input:
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7, 
    min_tracking_confidence=0.3
)

cap = cv2.VideoCapture(cv2.CAP_DSHOW)
while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        # If loading a video, use 'break' instead of 'continue'.
        continue

    # Flip the image horizontally for a later selfie-view display, and convert
    # the BGR image to RGB.
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
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

        # x_test = np.expand_dims(one_frame_feature, axis=0)
        x_test = np.expand_dims(one_frame_feature_shift, axis=0)
        result = model.predict(x_test)
        result_index = result[0].argmax()
        # print(result_index)
        cv2.putText(image, "Result:{} {}".format(result_index, labels[result_index]), (10, 30), 1, 2, (0, 0, 255), 2)

    cv2.imshow('MediaPipe Hands', image)
    if cv2.waitKey(5) & 0xFF == 27:
        break

hands.close()
cap.release()