import os
import random as rd
from threading import Thread

import cv2
import keras
import numpy as np

import gesture
import load
import model
import preproc

weights_path = "cnn3d.h5"
epochs = 30


class Augmentor(Thread):
    """
    Perform data augmentation on original dataset.
    """

    def __init__(self, original: np.ndarray, rotation_range: float = 20,
                 scale_range: float = 0.2, width_shift_range: float = 0.2,
                 height_shift_range: float = 0.2):
        """
        Constructor
        :param original: dataset directly read from files
            shape: [num_samples, num_chan, seq_len, img_height, img_width]
            dtype: numpy.uint8
        :param rotation_range: +- amount of rotation applied to data, in degrees
        :param scale_range: 1 +- scale factor applied to image
        :param width_shift_range: proportion of original width to apply shift in X axis
        :param height_shift_range: proportion of original height to apply shift in Y axis
        """
        super().__init__()
        self.original = original
        self.rotation_range = rotation_range
        assert scale_range < 1
        self.scale_range = scale_range
        self.width_shift_range = width_shift_range * original.shape[4]
        self.height_shift_range = height_shift_range * original.shape[3]
        self.rng = rd.Random()
        self.result = np.ndarray([len(original), model.input_length, model.input_height,
                                  model.input_width, 2], dtype=np.float32)

    def run(self) -> None:
        for sample_idx in range(len(self.original)):
            self.result[sample_idx] = self._transform_one_sample(self.original[sample_idx])

    def _transform_one_sample(self, sample: np.ndarray) -> np.ndarray:
        """
        Apply random transformation to one sample in the dataset.
        :param sample: channel-first gesture sequence
            shape: [num_chan, seq_len, img_height, img_width]
            dtype: numpy.uint8
        :return: transformed channel-last gesture sequence
            shape: [seq_len, img_height, img_width, num_chan]
            dtype: numpy.float32
        """
        # Compute common transformation matrix
        rot_deg = self.rng.uniform(-self.rotation_range, self.rotation_range)
        scale_factor = self.rng.uniform(1 - self.scale_range, 1 + self.scale_range)
        width_shift = self.rng.uniform(-self.width_shift_range, self.width_shift_range)
        height_shift = self.rng.uniform(-self.height_shift_range, self.height_shift_range)
        center = np.array([model.input_width, model.input_height]) / 2
        mat = cv2.getRotationMatrix2D(tuple(center), rot_deg, scale_factor)
        mat[0, 2] += width_shift
        mat[1, 2] += height_shift

        # Apply affine transformation to each frame in each channel
        transformed = np.ndarray(sample.shape, sample.dtype)
        for chan_idx in range(2):
            for frame_idx in range(sample.shape[1]):
                frame = sample[chan_idx][frame_idx].copy()
                frame = cv2.warpAffine(frame, mat, (model.input_width, model.input_height))
                transformed[chan_idx][frame_idx] = frame

        # Normalize data channel-wise and return
        return preproc.normalize_sample(transformed)


if __name__ == '__main__':
    nn = model.CNN3D()
    data_x, data_y = load.load_dataset("data")
    data_y = keras.utils.to_categorical(data_y, len(gesture.category_names))
    aug = Augmentor(data_x)
    aug.start()
    if os.path.exists(weights_path):
        print("Weight file is found, fine-tune on existing weights.")
        nn.load_weights(weights_path)
    for epoch_idx in range(epochs):
        print("Epoch %d/%d" % (epoch_idx, epochs))
        aug.join()
        aug_data_x = aug.result.copy()
        aug = Augmentor(data_x)  # a single thread object cannot be started more than once
        aug.start()  # run augmentation of next epoch concurrently with model training
        nn.fit(x=aug_data_x, y=data_y, batch_size=20)
    nn.save_weights(weights_path)
