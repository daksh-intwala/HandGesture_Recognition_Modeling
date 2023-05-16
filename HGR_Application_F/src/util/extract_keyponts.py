import mediapipe as mp
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import pandas as pd
import os
import math
from keras_preprocessing.sequence import pad_sequences
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)


class ExtractKeypoints:
    def __init__(self, video):
        self.video = video
        
    
    def getFrame(self, vid_path, sec):
        vid = cv2.VideoCapture(vid_path)
        vid.set(cv2.CAP_PROP_POS_MSEC, sec*1000)
        ret, image = vid.read()
        print("############ VID_PATH",vid_path)
        path = vid_path[:-9]+"output/"+vid_path[-9:-4]+f"{sec}.jpg"
        if ret:
            cv2.imwrite(path, image)

            return path
        
    def gen_frames(self, vid_path, vid_length=2, frames_per_sec=15):
        print("path: ",os.getcwd())
        n_iterations = vid_length * frames_per_sec
        frame_rate = 1 / frames_per_sec
        sec = 0
        frame_limit = 31
        dataset = np.empty((126)).reshape(1, -1)
        lst = [np.zeros(126)]

        for i in range(int(n_iterations)):
            sec += frame_rate
            sec = round(sec, 3)
            image = self.getFrame(vid_path, sec)
            if image:
                data_point = self.landmark_extractor(image)
            if len(data_point)<= frame_limit:
                dataset = np.append(dataset, data_point, axis=0)
            else :
                to_round = math.ceil(len(data_point)/frame_limit)
                dataset = np.append(data_point[::to_round])
        while len(dataset) <frame_limit:
            dataset = np.append(dataset,lst,axis=0)
        return dataset

    def landmark_extractor(self, img):
        mp_hands = mp.solutions.hands

        with mp_hands.Hands(static_image_mode=True,
                            max_num_hands=2,
                            min_detection_confidence=0.5) as hands:
            
                landmark = []

                image = cv2.flip(cv2.imread(img), 1)

                results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

                landmarks = results.multi_hand_landmarks
                
                if landmarks != None:
                    landmark_idx = mp_hands.HandLandmark.WRIST.value
                    landmark += (np.array([[lmk.landmark[landmark_idx].x, 
                                    lmk.landmark[landmark_idx].y,
                                    lmk.landmark[landmark_idx].z] for lmk in landmarks]).flatten() if landmarks else np.zeros(3)).tolist()

                    landmark_idx = mp_hands.HandLandmark.THUMB_CMC.value
                    landmark += (np.array([[lmk.landmark[landmark_idx].x, 
                                    lmk.landmark[landmark_idx].y, 
                                    lmk.landmark[landmark_idx].z] for lmk in landmarks]).flatten() if landmarks else np.zeros(3)).tolist()

                    landmark_idx = mp_hands.HandLandmark.THUMB_MCP.value
                    landmark += (np.array([[lmk.landmark[landmark_idx].x, 
                                    lmk.landmark[landmark_idx].y, 
                                    lmk.landmark[landmark_idx].z] for lmk in landmarks]).flatten() if landmarks else np.zeros(3)).tolist()       

                    landmark_idx = mp_hands.HandLandmark.THUMB_IP.value
                    landmark += (np.array([[lmk.landmark[landmark_idx].x, 
                                    lmk.landmark[landmark_idx].y, 
                                    lmk.landmark[landmark_idx].z] for lmk in landmarks]).flatten() if landmarks else np.zeros(3)).tolist()   

                    landmark_idx = mp_hands.HandLandmark.THUMB_TIP.value
                    landmark += (np.array([[lmk.landmark[landmark_idx].x, 
                                    lmk.landmark[landmark_idx].y, 
                                    lmk.landmark[landmark_idx].z] for lmk in landmarks]).flatten() if landmarks else np.zeros(3)).tolist()

                    landmark_idx = mp_hands.HandLandmark.INDEX_FINGER_MCP.value
                    landmark += (np.array([[lmk.landmark[landmark_idx].x, 
                                    lmk.landmark[landmark_idx].y, 
                                    lmk.landmark[landmark_idx].z] for lmk in landmarks]).flatten() if landmarks else np.zeros(3)).tolist()

                    landmark_idx = mp_hands.HandLandmark.INDEX_FINGER_PIP.value
                    landmark += (np.array([[lmk.landmark[landmark_idx].x, 
                                    lmk.landmark[landmark_idx].y, 
                                    lmk.landmark[landmark_idx].z] for lmk in landmarks]).flatten() if landmarks else np.zeros(3)).tolist()

                    landmark_idx = mp_hands.HandLandmark.INDEX_FINGER_DIP.value
                    landmark += (np.array([[lmk.landmark[landmark_idx].x, 
                                    lmk.landmark[landmark_idx].y, 
                                    lmk.landmark[landmark_idx].z] for lmk in landmarks]).flatten() if landmarks else np.zeros(3)).tolist()       

                    landmark_idx = mp_hands.HandLandmark.INDEX_FINGER_TIP.value
                    landmark += (np.array([[lmk.landmark[landmark_idx].x, 
                                    lmk.landmark[landmark_idx].y, 
                                    lmk.landmark[landmark_idx].z] for lmk in landmarks]).flatten() if landmarks else np.zeros(3)).tolist()   

                    landmark_idx = mp_hands.HandLandmark.MIDDLE_FINGER_MCP.value
                    landmark += (np.array([[lmk.landmark[landmark_idx].x, 
                                    lmk.landmark[landmark_idx].y, 
                                    lmk.landmark[landmark_idx].z] for lmk in landmarks]).flatten() if landmarks else np.zeros(3)).tolist()

                    landmark_idx = mp_hands.HandLandmark.MIDDLE_FINGER_PIP.value
                    landmark += (np.array([[lmk.landmark[landmark_idx].x, 
                                    lmk.landmark[landmark_idx].y, 
                                    lmk.landmark[landmark_idx].z] for lmk in landmarks]).flatten() if landmarks else np.zeros(3)).tolist()

                    landmark_idx = mp_hands.HandLandmark.MIDDLE_FINGER_DIP.value
                    landmark += (np.array([[lmk.landmark[landmark_idx].x, 
                                    lmk.landmark[landmark_idx].y, 
                                    lmk.landmark[landmark_idx].z] for lmk in landmarks]).flatten() if landmarks else np.zeros(3)).tolist()

                    landmark_idx = mp_hands.HandLandmark.MIDDLE_FINGER_TIP.value
                    landmark += (np.array([[lmk.landmark[landmark_idx].x, 
                                    lmk.landmark[landmark_idx].y, 
                                    lmk.landmark[landmark_idx].z] for lmk in landmarks]).flatten() if landmarks else np.zeros(3)).tolist()       

                    landmark_idx = mp_hands.HandLandmark.RING_FINGER_MCP.value
                    landmark += (np.array([[lmk.landmark[landmark_idx].x, 
                                    lmk.landmark[landmark_idx].y, 
                                    lmk.landmark[landmark_idx].z] for lmk in landmarks]).flatten() if landmarks else np.zeros(3)).tolist()   

                    landmark_idx = mp_hands.HandLandmark.RING_FINGER_PIP.value
                    landmark += (np.array([[lmk.landmark[landmark_idx].x, 
                                    lmk.landmark[landmark_idx].y, 
                                    lmk.landmark[landmark_idx].z] for lmk in landmarks]).flatten() if landmarks else np.zeros(3)).tolist()

                    landmark_idx = mp_hands.HandLandmark.RING_FINGER_DIP.value
                    landmark += (np.array([[lmk.landmark[landmark_idx].x, 
                                    lmk.landmark[landmark_idx].y, 
                                    lmk.landmark[landmark_idx].z] for lmk in landmarks]).flatten() if landmarks else np.zeros(3)).tolist()

                    landmark_idx = mp_hands.HandLandmark.RING_FINGER_TIP.value
                    landmark += (np.array([[lmk.landmark[landmark_idx].x, 
                                    lmk.landmark[landmark_idx].y, 
                                    lmk.landmark[landmark_idx].z] for lmk in landmarks]).flatten() if landmarks else np.zeros(3)).tolist()

                    landmark_idx = mp_hands.HandLandmark.PINKY_MCP.value
                    landmark += (np.array([[lmk.landmark[landmark_idx].x, 
                                    lmk.landmark[landmark_idx].y, 
                                    lmk.landmark[landmark_idx].z] for lmk in landmarks]).flatten() if landmarks else np.zeros(3)).tolist()       

                    landmark_idx = mp_hands.HandLandmark.PINKY_PIP.value
                    landmark += (np.array([[lmk.landmark[landmark_idx].x, 
                                    lmk.landmark[landmark_idx].y, 
                                    lmk.landmark[landmark_idx].z] for lmk in landmarks]).flatten() if landmarks else np.zeros(3)).tolist()   

                    landmark_idx = mp_hands.HandLandmark.PINKY_DIP.value 
                    landmark += (np.array([[lmk.landmark[landmark_idx].x, 
                                    lmk.landmark[landmark_idx].y, 
                                    lmk.landmark[landmark_idx].z] for lmk in landmarks]).flatten() if landmarks else np.zeros(3)).tolist()

                    landmark_idx = mp_hands.HandLandmark.PINKY_TIP.value
                    landmark += (np.array([[lmk.landmark[landmark_idx].x, 
                                    lmk.landmark[landmark_idx].y, 
                                    lmk.landmark[landmark_idx].z] for lmk in landmarks]).flatten() if landmarks else np.zeros(3)).tolist()

                if len(landmark) == 126:
                    row = np.around(landmark, decimals=5).reshape(1, -1)
                else:
                    row = np.hstack((np.array(landmark), np.zeros(126-len(landmark)))).reshape(1, -1)
                return row

    def make_csv(self):
        action_test = np.empty((126)).reshape(1, -1)
        action_test = np.delete(action_test,np.s_[:],axis=0)

        int_data = self.gen_frames(self.video)
        action_padded = pad_sequences(int_data, padding='post', maxlen=126, dtype='float32')
        action_test = np.append(action_test, action_padded, axis=0)
        logger.info("action_test",len(action_test))

        pd.DataFrame(action_test).to_csv('user_clip.csv')  

        return 'user_clip.csv'

if __name__=="__main__":
    prd = ExtractKeypoints('/Users/dakshintwala/HGR_Application/src/99999.mp4')
    path = prd.make_csv()
    print(path)