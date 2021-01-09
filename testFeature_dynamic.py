import cv2, os, sys, time, h5py
import mediapipe as mp
import numpy as np

video_folder_path = "Video/滑鼠右鍵"

save_filename = "data_dynamic.h5"

fps = 60
timeout_ms = int(1000.0/fps)
####my label param####

label_basics = ["MouseLeft", "MouseRight", "MouseMove",
                "PalmUp", "PalmDown", "PalmLeft", "PalmRight", "PalmMove",
                "FingerClockwise", "FingerCounterclockwise", "FingerMove"]
label_name = label_basics[1]   
######################

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

video_paths = []
dataset_names = []
for path in os.listdir(video_folder_path):
    full_path = os.path.join(video_folder_path, path)
    if os.path.isfile(full_path):
        video_paths.append(full_path)


hands = mp_hands.Hands(
    static_image_mode=False,
    # static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.7, 
    min_tracking_confidence=0.3
)

end_flag = False
video_features = []

file_cnt = 0
for video_path in video_paths:
    filename = os.path.basename(video_path)
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap_type = cap.get(cv2.CAP_PROP_BACKEND)
    all_frame_features = []
    frame_cnt = 0
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
            if ((fps == 60) and (frame_cnt % 2 == 0)) or fps == 30:
                all_frame_features.append(one_frame_feature)
        frame_cnt += 1
        cv2.putText(image, "Cnt:{}, Frame Cnt:{}".format(file_cnt, frame_cnt), (10, 50), 1, 2, (0, 0, 255), 2)
        cv2.putText(image, "File:{}".format(filename), (10, 30), 1, 2, (0, 0, 255), 2)
        cv2.imshow('MediaPipe Hands', image)
        if cv2.waitKey(timeout_ms) & 0xFF == 27:
            end_flag = True
            break
    if end_flag:
        break
    all_frame_features = np.array(all_frame_features)
    video_features.append(all_frame_features)
    dataset_name = "{}_{}".format(label_name, int(time.time()))
    dataset_names.append(dataset_name)
    file_cnt += 1
hands.close()
cap.release()

with h5py.File(save_filename, 'a') as h5f:
    if label_name not in h5f:
        group = h5f.create_group(label_name)
    else:
        group = h5f[label_name]

    for i, feature in enumerate(video_features):
        dataset_name = dataset_names[i]
        group[dataset_name] = feature
    print("end save {}".format(save_filename))
