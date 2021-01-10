import cv2, os, sys, time, h5py
import mediapipe as mp
import numpy as np
import tensorflow as tf
from utils import *
import matplotlib.pyplot as plt
from PPT_Controler import PPTControler

func_window = (170, 100, 300, 280) ## x, y , w, h
ppt_is_playing = False
dynamic_labels = ['MouseLeft', 'MouseMove', 'MouseRight']
static_labels = ['VirtualFive', 'VirtualFour', 'VirtualGood', 
          'VirtualOK', 'VirtualOne', 'VirtualRotate', 
          'VirtualStone', 'VirtualThree', 'VirtualTwo']

avg_cnt = 40
model_static = tf.keras.models.load_model('model_9.h5')
model_dynamic = tf.keras.models.load_model('model_dynamic_mouse_hs.h5')

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7, 
    min_tracking_confidence=0.7
)

ppt = PPTControler()

cap = cv2.VideoCapture(1)
static_gesture_cnt = 0
result_index_conv = -1
result_index_conv_edge = -1
previous_static_gesture = -1
previous_dynamic_gesture = -1
previous_index_hand_point = (-1, -1)

total_frame = []
conv_fifo_left, conv_fifo_right = [], []
mode = ""
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

    # cv2.rectangle(image, (func_window[0], func_window[1]), (func_window[0]+func_window[2], 
    #                     func_window[1]+func_window[3]), (0, 0, 255), 2)

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

        if len(total_frame) < avg_cnt:
            total_frame.append(one_frame_feature_shift)
        else:
            total_frame.pop(0)
            total_frame.append(one_frame_feature_shift)
        total_frame_np = np.array(total_frame)
        x_test_total = np.expand_dims(total_frame_np, axis=0)

        # x_test = np.expand_dims(one_frame_feature, axis=0)
        x_test = np.expand_dims(one_frame_feature_shift, axis=0)
        result = model_static.predict(x_test)
        result_index = result[0].argmax()

        result_dynamic = model_dynamic.predict(x_test_total)
        if len(conv_fifo_left) < 15:
            conv_fifo_left.append(result_dynamic[0][0])
        else:
            conv_fifo_left.pop(0)
            conv_fifo_left.append(result_dynamic[0][0])

        if len(conv_fifo_right) < 15:
            conv_fifo_right.append(result_dynamic[0][2])
        else:
            conv_fifo_right.pop(0)
            conv_fifo_right.append(result_dynamic[0][2])

        left_conv = predictFilter(conv_fifo_left)
        right_conv = predictFilter(conv_fifo_right)
        
        previous_dynamic_gesture = result_index_conv
        if not(left_conv or right_conv):
            result_index_conv = 1
        elif left_conv:
            result_index_conv = 0
        elif right_conv:
            result_index_conv = 2
        else:
            result_index_conv = 1
        if previous_dynamic_gesture != result_index_conv: ## rising edge
            result_index_conv_edge = result_index_conv
        else:
            result_index_conv_edge = 1
        ##gesture control logic
        # if handInROI(one_frame_feature, image, func_window):
        if True:
            # cv2.rectangle(image, (func_window[0], func_window[1]), (func_window[0]+func_window[2], 
            #             func_window[1]+func_window[3]), (0, 255, 0), 2)
            if previous_static_gesture != result_index:
                previous_static_gesture = result_index
                static_gesture_cnt = 1
            else:
                static_gesture_cnt += 1
            cv2.putText(image, "Same Gesture Cnt:{}".format(static_gesture_cnt), (10, 50), 1, 2, (0, 0, 255), 2)
            if static_gesture_cnt >= 3:
                if static_labels[previous_static_gesture] == "VirtualGood":
                    static_gesture_cnt_thresh = 20
                elif static_labels[previous_static_gesture] == "VirtualOne":
                    static_gesture_cnt_thresh = 15
                elif static_labels[previous_static_gesture] == "VirtualTwo":
                    static_gesture_cnt_thresh = 5
                else:
                    static_gesture_cnt_thresh = 15
            else:
                static_gesture_cnt_thresh = 15
            
            if static_gesture_cnt >= static_gesture_cnt_thresh:
                if previous_static_gesture != -1:
                    if static_labels[previous_static_gesture] == "VirtualGood":
                        thumb = getCoord(one_frame_feature_shift[4])
                        little_finger = getCoord(one_frame_feature_shift[17])
                        x_len = thumb[0] - little_finger[0]
                        y_len = thumb[1] - little_finger[1]
                        if abs(y_len) > abs(x_len): ##vertical
                            if y_len <= 0: ## good
                                ppt.fullScreen()
                                cv2.putText(image, "Start playing slide!", (10, 70), 1, 2, (0, 0, 255), 2)
                            else: ## bad
                                ppt.exitFullScreen()
                                cv2.putText(image, "End playing slide!", (10, 70), 1, 2, (0, 0, 255), 2)

                        elif abs(x_len) > abs(y_len): ##horizon
                            if x_len <= 0: ##left
                                ppt.prePage()
                            else: ##right
                                ppt.nextPage()
                        else:
                            pass

                        print(thumb, little_finger)
                        # ppt.fullScreen()
                        previous_static_gesture = -1
                        static_gesture_cnt = 0

                    elif static_labels[previous_static_gesture] == "VirtualTwo":
                        mode = "VirtualTwo"
                        
                    elif static_labels[previous_static_gesture] == "VirtualOne":
                        mode = "VirtualOne"
                        ppt.activateLaserPointer()
                        previous_static_gesture = -1
                        static_gesture_cnt = 0
            
            if mode == "VirtualTwo":
                if dynamic_labels[result_index_conv_edge] == "MouseLeft":
                    print("MouseLeft")
                    ppt.prePage()
                elif dynamic_labels[result_index_conv_edge] == "MouseRight":
                    print("MouseRight")
                    ppt.nextPage()
                else:
                    # print("MouseMove")
                    pass
            elif mode == "VirtualOne":
                index_finger = getCoord(one_frame_feature[8])
                # print(previous_index_hand_point, index_finger)

                if previous_index_hand_point[0] == -1:
                    pass
                else:
                    previous_index_hand_point_extract = (int(previous_index_hand_point[0]*1920/640), int(previous_index_hand_point[1]*1920/480))
                    index_finger_extract = (int(index_finger[0]*1920/640), int(index_finger[1]*1920/480))
                    moveFromTo(previous_index_hand_point_extract, index_finger_extract)
                    
                previous_index_hand_point = index_finger
                
        else:
            previous_static_gesture = -1
            static_gesture_cnt = 0
        cv2.putText(image, "Result:{} {}".format(result_index, static_labels[result_index]), (10, 30), 1, 2, (0, 0, 255), 2)
    else:
        previous_static_gesture = -1
        static_gesture_cnt = 0
    cv2.imshow('MediaPipe Hands', image)
    if cv2.waitKey(5) & 0xFF == 27:
        break

hands.close()
cap.release()
