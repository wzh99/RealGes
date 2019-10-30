from threading import Thread
from typing import List

import cv2
import keras
import numpy as np

import gesture
import model
import preproc
from capture import Recorder, Camera


class Recognizer(Thread):
    """
    Recognizer thread that runs in parallel with recorder thread
    """

    def __init__(self, mdl: keras.Model, seq: List[np.ndarray]):
        """
        Constructor
        :param mdl: a Keras model used to recognize recorded gesture
        :param seq: [depth_seq, gradient_seq] recorded
        """
        super().__init__()
        self.model = mdl
        self.seq = seq

    def run(self) -> None:
        # Rescale image sequence
        resized_seq = []
        for channel_seq in self.seq:
            resized_channel = []
            for img in channel_seq:
                resized_img = cv2.resize(img, (model.input_width, model.input_height))
                resized_channel.append(resized_img)
            resized_seq.append(np.array(resized_channel))

        # Normalize sequence
        normalized = preproc.normalize_sequence(resized_seq)

        # Pass to model for prediction result
        result = self.model.predict(np.array([normalized]))
        index = np.argmax(result[0])
        print("gesture: %s" % gesture.category_names[index])


if __name__ == '__main__':
    cnn = model.CNN3D()
    cnn.load_weights("cnn3d.h5")
    rec = Recorder(Camera(), callback=lambda seq: Recognizer(cnn, seq).run())
    rec.record()
