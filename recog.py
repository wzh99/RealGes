import os
from threading import Thread
from typing import List

import cv2
import keras
import numpy as np

import gesture
import model
import preproc
from capture import Recorder


class Recognizer(Thread):
    """
    Recognizer thread that runs in parallel with recorder thread
    """

    def __init__(self, hrn: keras.Model, lrn: keras.Model, sample: List[np.ndarray]):
        """
        Constructor
        :param hrn: a Keras model used to recognize recorded gesture
        :param sample: [depth_chan, grad_chan]
        """
        super().__init__()
        self.hrn = hrn
        self.lrn = lrn
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
        data = np.array([preproc.normalize_sample(data)])

        # Pass to model for prediction result
        result = self.hrn.predict(data) * self.lrn.predict(data)
        index = np.argmax(result[0])
        print("gesture: %s" % gesture.category_names[index])


if __name__ == '__main__':
    hrn_spec = model.network_spec["lrn"]
    hrn_model = hrn_spec["init"]()
    if not os.path.exists(hrn_spec["path"]):
        raise RuntimeError("HRN weight file not found.")
    hrn_model.load_weights(hrn_spec["path"])
    lrn_spec = model.network_spec["lrn"]
    lrn_model = lrn_spec["init"]()
    if not os.path.exists(lrn_spec["path"]):
        raise RuntimeError("LRN weight file not found.")
    lrn_model.load_weights(lrn_spec["path"])
    rec = Recorder(callback=lambda seq: Recognizer(hrn_model, lrn_model, seq).run())
    rec.record()
