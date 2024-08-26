import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import LSTM
import streamlit as st

labels = np.array(['FALL', 'LYING', 'SIT', 'STAND', 'MOVE'])

n_time_steps = 25
mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils

def custom_lstm(*args, **kwargs):
    kwargs.pop('time_major', None)
    return LSTM(*args, **kwargs)

model = tf.keras.models.load_model('bro.h5', custom_objects={'LSTM': custom_lstm})

def make_landmark_timestep(results):
    c_lm = []
    for id, lm in enumerate(results.pose_landmarks.landmark):
        c_lm.append(lm.x)
        c_lm.append(lm.y)
        c_lm.append(lm.z)
        c_lm.append(lm.visibility)
    return c_lm

def draw_landmark_on_image(mpDraw, results, img, label):
    mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
    for id, lm in enumerate(results.pose_landmarks.landmark):
        h, w, c = img.shape
        cx, cy = int(lm.x * w), int(lm.y * h)
        if label != "FALL":
            cv2.circle(img, (cx, cy), 5, (0, 255, 0), cv2.FILLED)
        else:
            cv2.circle(img, (cx, cy), 5, (0, 0, 255), cv2.FILLED)
    return img

def draw_class_on_image(label, img):
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (10, 30)
    fontScale = 1
    fontColor = (0, 255, 0)
    thickness = 2
    lineType = 2
    cv2.putText(img, label,
                bottomLeftCornerOfText,
                font,
                fontScale,
                fontColor,
                thickness,
                lineType)
    return img

def detect(model, lm_list):
    lm_list = np.array(lm_list)
    lm_list = np.expand_dims(lm_list, axis=0)
    results = model.predict(lm_list)
    if results[0][0] >= 0.5: 
        label = labels[0]
    elif results[0][1] >= 0.5:
        label = labels[1]  
    elif results[0][2] >= 0.5:
        label = labels[2]
    elif results[0][3] >= 0.5:
        label = labels[3]
    elif results[0][4] >= 0.5:
        label = labels[4]
    else:
        label = "NONE DETECTION"
    return label

def main():
    st.title("Pose Detection and Classification")
    
    run_type = st.sidebar.selectbox("Select input type", ("Camera", "Video File"))
    
    if run_type == "Camera":
        cap = cv2.VideoCapture(0)
    else:
        video_file = st.sidebar.file_uploader("Upload a video", type=["mp4", "mov", "avi"])
        if video_file is not None:
            # Temporarily save the uploaded video to disk to pass to cv2.VideoCapture
            with open("temp_video.mp4", "wb") as f:
                f.write(video_file.read())
            cap = cv2.VideoCapture("temp_video.mp4")
        else:
            st.write("Please upload a video file.")
            return
    
    stframe = st.empty()
    label = 'Starting...'
    lm_list = []

    while cap.isOpened():
        success, img = cap.read()
        if not success:
            break
        
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = pose.process(imgRGB)
        
        if results.pose_landmarks:
            c_lm = make_landmark_timestep(results)
            img = draw_landmark_on_image(mpDraw, results, img, label)
            img = draw_class_on_image(label, img)
            lm_list.append(c_lm)
            if len(lm_list) == n_time_steps:
                label = detect(model, lm_list)
                lm_list = []
        
        stframe.image(img, channels="BGR")
        
        if cv2.waitKey(1) == ord('q'):
            break
    
    cap.release()

if __name__ == '__main__':
    main()
