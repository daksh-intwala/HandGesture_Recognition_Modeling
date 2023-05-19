# HandGesture_Recognition_Modeling
### Abstract:
In recent times, hand gesture recognition has attained considerable attention attributing to its potential application in diverse domains such as human computer interactions, robotics, and virtual reality. This report attempts to contribute to the development and evaluation of hand gesture recognition through deep learning techniques. 

The proposed model uses a special recurrent neural network that solves long term dependency problems by taking a video input feed as input. The model is trained on a large newly created dataset consisting of short videos allowing it to learn different gestures effectively. 
Furthermore, the report discusses the shortcomings and future scope of the potential described in our methodology.

### Aim:
The goal of this project is to build a hand gesture recognition system using OpenCV, Google’s MediaPipe, and an LSTM model. The goal of the system is to recognize hand gestures, such as "hello" and "goodbye," from video clips. The first step in the system is to record short video clips of 2 seconds in which a person performs the action. OpenCV is subsequently leveraged to extract images from the video clips at 15 frames per second. Landmarks are extracted from these images using the MediaPipe. These landmarks are coordinates of different location on the hand that can be used to identify the gesture. The extracted landmarks are then passed to an LSTM, which is a type of neural network that can learn long-term dependencies. The LSTM is trained on a dataset of images and their corresponding labels. The labels indicate which gesture is being performed in each image. Once the LSTM is trained, it can be used to recognize hand gestures from new images. The hand gesture recognition system that we built is able to recognize a variety of hand gestures with a reasonable degree of accuracy.

### Literature Survey
Earlier approaches at hand gesture recognition included following steps: hand segmentation, hand frame tracking, feature extraction, classification. 
1. Suresh, Chandrasekhar, and Dinesh presented an analysis and evaluation of widely recognized semantic segmentation models for the purpose of hand region segmentation. Additionally, an ensemble approach is employed to fuse the segmented RGB, enabling hand gesture classification based on probability scores. The experimental findings demonstrate that the newly proposed framework, Semantic Segmentation and Ensemble Classification (SSEC), is well-suited for static hand gesture recognition and achieves an F1-score of 88.91% on the test dataset.
2. Ankita Wadhawan and Parteek Kumar proposed a sign language recognition system in which the training process is based on convolutional neural networks, where preprocessed sign images are inputted into the classifier to assign them to the appropriate categories. To train the classifier, a dataset containing various ISL (Indian Sign Language) gestures was utilized. The system achieved high training and validation accuracy, specifically 99.76% and 98.35% respectively, using the RMSProp optimizer. Furthermore, they observed that the SGD optimizer surpasses the performance of Adam, RMSProp, and other optimizers, achieving training and validation accuracy of 99.90% and 98.70% respectively, on a grayscale image dataset. 
3. The paper used different architectures for training a hand gesture recognition system. These include 3D convolutional neural networks (I3D), convolutional networks coupled with recurrent neural networks, and proposed a novel method called temporal graph convolutional network (TGCN). Trained on a dataset of 100 gestures, they achieved a 5 fold cross validation accuracy of 84% using I3D, 55% using VGG-GRU, and 79% using the TGCN.

### Dataset 
Link to Data - (https://drive.google.com/drive/folders/1s-FU0QDrfY3GNw6dvZncJWxrz58zjy6u?usp=share_link)

### Conclusion
In conclusion, through this project, we were able to develop a structured implementation which can detect an action under limited scope of data which is performed in ASL. As a result, we were successful in interpreting and processing a 3D video clip, parsing it into 2D frames and feeding it to an LSTM architecture in sequence while not disrupting the order of the frames. The LSTM model trained is capable enough to predict one of the nine ASL action present in the dataset with 61.11% of accuracy. Outcome of this project is that a user, unaware of ASL, can use this developed technology to understand and communicate with an impaired person who is supposed to have ASL as the sole way of communication. 

### Future Scope
Currently, the LSTM Model trained in this project can be improvised with increased accuracy if we can collect 5x times the data for each action with maintaining the variance in the data while we feed into LSTM model. The challenge faced by our team was that it is a complex process to maintain the data variance at the same time maintaining the correct sequence of the input frames. In future, solving the variance can bring noticeable improvement in the performance of the model. 

Moving forward, we can use the same implementation developed in this project and predict actions and words for a long video clip which represents a sentence. This will result into predictions of verbs and nouns. The objective should be to collect all these predictions, apply semantic analysis on it, summarize the data, and feed it into a Natural Language Generation architecture to form a sensible sentence. In this way, we are converting ASL actions into text. After this point, we can use text-to-speech algorithm to give voice to the user who actually doesn’t possess ability to speak. This generated sentence can also be used as a subtitle text for ASL video. 



