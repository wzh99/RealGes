import os
from threading import Thread
from typing import List

import cv2
import keras
import numpy as np

import gesture
import model
import preproc
import train
from capture import Recorder


class Recognizer(Thread):
    """
    Recognizer thread that runs in parallel with recorder thread
    """

    def __init__(self, mdl: keras.Model, sample: List[np.ndarray]):
        """
        Constructor
        :param mdl: a Keras model used to recognize recorded gesture
        :param sample: [depth_chan, grad_chan]
        """
        super().__init__()
        self.model = mdl
        self.sample = sample

    def run(self) -> None:
        # Rescale image sequence
        data = np.ndarray([2, model.input_length, model.input_height, model.input_width])
        for chan_idx in range(2):
            chan = np.ndarray([0, model.input_height, model.input_width])
            for frame_idx in range(len(self.sample[chan_idx])):
                frame = self.sample[chan_idx][frame_idx]
                resized = cv2.resize(frame, (model.input_width, model.input_height))
                chan = np.append(chan, [resized], axis=0)
            data[chan_idx] = preproc.temporal_resample(chan, model.input_length)

        # Normalize sequence
        data = preproc.normalize_sample(data)

        # Pass to model for prediction result
        result = self.model.predict(np.array([data]))
        index = np.argmax(result[0])
        print("gesture: %s" % gesture.category_names[index])


if __name__ == '__main__':
    nn = model.CNN3D()
    if not os.path.exists(train.weights_path):
        raise RuntimeError("Weight file not found. Cannot run gesture recognition program.")
    nn.load_weights(train.weights_path)
    rec = Recorder(callback=lambda seq: Recognizer(nn, seq).run())
    rec.record()
