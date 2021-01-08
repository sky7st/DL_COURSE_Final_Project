import cv2, os, sys, time, h5py
import mediapipe as mp
import numpy as np

####my label param####

label_basics = ["VirtualOne", "VirtualTwo", "VirtualThree",
                "VirtualFour","VirtualScissors", "VirtualPaper",
                "VirtualStone","VirtualGood", "VirtualOK",
                "VirtualRotate","Other"]
# label_name = "VirtualMouse"    ## 2 finger
# label_name = "VirtualSelect"   ## 1 finger
label_name = label_basics[0]     ## other (background)

######################

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
cam_h = 480
cam_w = 640
# For webcam input:
hands = mp_hands.Hands(
    static_image_mode=False,
    # static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.7, 
    min_tracking_confidence=0.3
)
cap = cv2.VideoCapture(cv2.CAP_DSHOW)
all_frame_features = []
cnt = 0
def drawData(img, data):
    for i in range(data.shape[0]):
        x = int(data[i][0] * cam_w)
        y = int(data[i][1] * cam_h)
        img = cv2.circle(img, (x, y), radius=0, color=(0, 0, 255), thickness=4)
    return img
def calShift(data):
    data_copy = data.copy()
    shift_arr = np.argmin(data, axis=0)
    data_copy[:,0] -= data[shift_arr[0],0] ##X
    data_copy[:,1] -= data[shift_arr[1],1] ##Y
    data_copy[:,2] -= data[shift_arr[2],2] ##Z
    return data_copy
while cap.isOpened():
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(fps)
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
        # print(results.multi_handedness)
        # for hand_landmarks in results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]

        mp_drawing.draw_landmarks(image, hand_landmarks)
        
        for i, data_point in enumerate(hand_landmarks.landmark):
            one_frame_feature.append(
                np.array([data_point.x, data_point.y, data_point.z])
            )
        one_frame_feature = np.array(one_frame_feature)

        one_frame_feature_shift = calShift(one_frame_feature)
        image = drawData(image, one_frame_feature_shift)
        all_frame_features.append(one_frame_feature_shift)
        cnt += 1
    cv2.putText(image, "Cnt:{}".format(cnt), (10, 30), 1, 2, (0, 0, 255), 2)

    cv2.imshow('MediaPipe Hands', image)
    if cv2.waitKey(5) & 0xFF == 27:
        break
hands.close()
cap.release()

all_frame_features = np.array(all_frame_features)
shape = all_frame_features.shape
dataset_name = "{}_{}".format(label_name, int(time.time()))

# with h5py.File('data1.h5', 'a') as h5f:
#     h5f.create_dataset(dataset_name, data=all_frame_features)
