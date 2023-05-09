# HandGesture_Recognition_Modeling
### Aim:
We aim to develop an image classification model to perform live streaming sign language recognition which converts sign language into text format using image recognition techniques through DNN model and deliver text phrases to indicate context in the live video. To test the trained model, we plan to use the confusion matrix as a performance matrix where we would evaluate F1, Recall and Precision scores to determine model reliability and validate the results with cross validation scores using GridSearchCV. Using OpenCV to create the DNN network is our goal.

### Literature Survey
Earlier approaches at hand gesture recognition included following steps: hand segmentation, hand frame tracking, feature extraction, classification. 
1. Suresh, Chandrasekhar, and Dinesh presented an analysis and evaluation of widely recognized semantic segmentation models for the purpose of hand region segmentation. Additionally, an ensemble approach is employed to fuse the segmented RGB, enabling hand gesture classification based on probability scores. The experimental findings demonstrate that the newly proposed framework, Semantic Segmentation and Ensemble Classification (SSEC), is well-suited for static hand gesture recognition and achieves an F1-score of 88.91% on the test dataset.
2. Ankita Wadhawan and Parteek Kumar proposed a sign language recognition system in which the training process is based on convolutional neural networks, where preprocessed sign images are inputted into the classifier to assign them to the appropriate categories. To train the classifier, a dataset containing various ISL (Indian Sign Language) gestures was utilized. The system achieved high training and validation accuracy, specifically 99.76% and 98.35% respectively, using the RMSProp optimizer. Furthermore, they observed that the SGD optimizer surpasses the performance of Adam, RMSProp, and other optimizers, achieving training and validation accuracy of 99.90% and 98.70% respectively, on a grayscale image dataset. 
3. The paper used different architectures for training a hand gesture recognition system. These include 3D convolutional neural networks (I3D), convolutional networks coupled with recurrent neural networks, and proposed a novel method called temporal graph convolutional network (TGCN). Trained on a dataset of 100 gestures, they achieved a 5 fold cross validation accuracy of 84% using I3D, 55% using VGG-GRU, and 79% using the TGCN.
### Status 
1. Divided single video clip into optimal frames 0.067 setting. - Jaydeep
2. Dynamically extracted the data from Kaggle, labelled videos using JSON config, created frames for multiple videos related to action - Kratika
3. Ability to extract landmark and draw landmark dynamically for further usecase, created model pipeline with mediapipe HandLandmarks - Daksh
4. Landmark dataset creation - Jaydeep
5. LSTM - Kratika
