import keras
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

import time

import model
import gesture
import data
import preproc

class ModelTester:
    def __init__(self, m1_name: str, m2_name: str):
        """
        Constructor
        :param m1_name: name of first model to test
        :param m2_name: name of second model to test
        """
        self.m1, self.m2 = model.load_two_models(m1_name, m2_name)

    def test(self, data_x: np.ndarray, data_y: np.ndarray) -> None:
        """
        Test models on given data.
        :param data_x: data to be predicted
        :param data_y: truth of category index
        """
        # Normalize data for prediction
        norm_x = np.array([preproc.normalize_sample(sp) for sp in data_x])

        # Predict gestures
        begin = time.time()
        if self.m1 == self.m2:
            result = self.m1.predict(norm_x)
        else:
            result = self.m1.predict(norm_x) * self.m2.predict(norm_x)
        end = time.time()

        # Display results
        print("Time:", (end - begin) / len(data_y))
        pred_y = np.argmax(result, axis=1)
        print("Accuracy:", (pred_y == data_y).tolist().count(True) / len(data_y))

        # Plot confusion matrix
        cm = confusion_matrix(data_y, pred_y)
        cm = cm.astype(np.float) / cm.sum(axis=1)[:, np.newaxis]
        plt.figure(figsize=(6, 5.5))
        plt.imshow(cm, interpolation='nearest')
        title_name = self.m1.__class__.__name__ if self.m1 == self.m2 else "%s & %s" % (
            self.m1.__class__.__name__, self.m2.__class__.__name__
        )
        plt.title("%s Gesture Confusion Matrix" % title_name)
        plt.colorbar()
        num_local = np.array(range(len(gesture.category_names)))    
        plt.xticks(num_local, gesture.category_names, rotation=90)
        plt.yticks(num_local, gesture.category_names)
        plt.ylabel('Truth')    
        plt.xlabel('Prediction')
        plt.subplots_adjust(left=0.15, bottom=0.25, right=0.98, top=0.88)
        plt.show()


if __name__ == "__main__":
    """
    Model       Time    Accuracy
    HRN + LRN   0.261   0.9896
    HRN         0.210   0.9618
    LRN         0.047   0.9757
    C3D         1.049   0.9410
    """
    data_x, data_y = data.from_hdf5("test_data.h5")
    tester = ModelTester("lrn", "lrn")
    tester.test(data_x, data_y)
