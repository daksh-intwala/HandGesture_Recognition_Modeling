import pandas as pd
import numpy as np
import os
import tensorflow
from tensorflow.keras.models import model_from_json
import logging
import matplotlib as pyplot


logger = logging.getLogger()
logger.setLevel(logging.INFO)

class Predict:
    def __init__(self, data_path, model_path, model_weight):
        self.data = data_path
        self.model_path = model_path
        self.model_weight = model_weight
        # self.x_test = np.ndarray()
        # self.y_test = np.ndarray()

    def test_data(self):
        x_test = pd.read_csv(self.data)
        x_test = x_test.iloc[: , :-1].to_numpy()
        print("x_test SHAPE ####",x_test.shape)
        x_test = x_test.reshape(1,31,126)
        logger.info(f"X test Shape-  {x_test.shape}")

        # y_test = []
        # for label in range(1,10):
        #     for instance in range(0,54):
        #         y_test = np.append(y_test,instance).astype(int)
        # y_test = y_test.reshape(54,9)
        # logger.info(f"X test Shape-  {y_test.shape}")

        # self.x_test =x_test
        # self.y_test = y_test
        # return x_test, y_test
        return x_test

    def load_model(self):
        json_file = open(self.model_path, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)

        # load weights into new model
        loaded_model.load_weights(self.model_weight)
        logger.info(f"Loaded model - {self.model_path} with weights {self.model_weight}")

        return loaded_model
    
    # # Re-evaluate the model
    # def re_evaluate_model(self):
    #     loaded_model  = self.load_model()
    #     x_test, y_test = self.test_data()
    
    #     loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    #     loss, acc = loaded_model.evaluate(x_test, y_test, verbose=2)
    #     logger.info("Restored model, accuracy: {:5.2f}%".format(100 * acc))

    #     return loss, acc

    def predict_action(self):
        loaded_model = self.load_model()

        res = loaded_model.predict(self.test_data())

        actions = ['1','2','3','4','5','6','7','8','9']
        result = actions[np.argmax(res)]
        labels = ['Chair','Computer','Drink','No','Help','Father','Thank You','How','Hello']
        
        # predicted_output = []
        # for i in range(0,54):
        #     logger.info(f"prediction loop for x_test: {actions[np.argmax(res[i])]}")
        #     predicted_output.append(actions[np.argmax(res[i])])
        
        return labels[actions.index(result)]
    
    # def plot_prediction(self):
    #     lst2 = [item[0] for item in self.test_y]
    #     pyplot.plot(lst2, self.predict_action())
    #     pyplot.show()
    

if __name__ == "__main__":
    data_path="/Users/dakshintwala/HGR_Application/user_clip.csv"
    model_path="/Users/dakshintwala/HGR_Application/model_11.json"
    model_weight="/Users/dakshintwala/HGR_Application/model_11.h5"
    pred = Predict(data_path,model_path,model_weight)
    result=pred.predict_action()
    print(result)
    