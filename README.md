# HandGesture_Recognition_Modeling
### Aim:
We aim to develop an image classification model to perform live streaming sign language recognition which converts sign language into text format using image recognition techniques through DNN model and deliver text phrases to indicate context in the live video. To test the trained model, we plan to use the confusion matrix as a performance matrix where we would evaluate F1, Recall and Precision scores to determine model reliability and validate the results with cross validation scores using GridSearchCV. Using OpenCV to create the DNN network is our goal.


### Status 
1. Divided single video clip into optimal frames 0.067 setting. - Jaydeep
2. Dynamically extracted the data from Kaggle, labelled videos using JSON config, created frames for multiple videos related to action - Kratika
3. Ability to extract landmark and draw landmark dynamically for further usecase, created model pipeline with mediapipe HandLandmarks - Daksh
4. Landmark dataset creation - Jaydeep
5. LSTM - Kratika
