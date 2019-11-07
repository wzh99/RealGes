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
epochs = 10


class Augmentor(Thread):
    """
    Perform data augmentation on original dataset.
    """

    def __init__(self, original: np.ndarray, rotation_range: float = 30,
                 scale_ratio: float = 0.2, width_shift_ratio: float = 0.2,
                 height_shift_ratio: float = 0.2, clip_ratio: float = 0.2,
                 temporal_elastic_range: float = 0.2):
        """
        Constructor
        :param original: dataset directly read from files
            shape: [num_samples, num_chan, seq_len, img_height, img_width]
            dtype: numpy.uint8
        :param rotation_range: +- amount of rotation applied to data, in degrees
        :param scale_ratio: 1 +- scale factor applied to image
        :param width_shift_ratio: proportion of original width to apply shift in X axis
        :param height_shift_ratio: proportion of original height to apply shift in Y axis
        :param clip_ratio: proportion of frames to be clipped
        :param temporal_elastic_range: 1 +- power in temporal elastic deformation curve
        """
        super().__init__()
        self.original = original
        self.rotation_range = rotation_range
        assert scale_ratio < 1
        self.scale_range = scale_ratio
        self.width_shift_range = width_shift_ratio * original.shape[4]
        self.height_shift_range = height_shift_ratio * original.shape[3]
        assert clip_ratio < 1
        self.max_clip_len = clip_ratio * original.shape[2]
        assert temporal_elastic_range < 1
        self.temporal_elastic_range = temporal_elastic_range
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
        result = np.ndarray(sample.shape, sample.dtype)
        seq_len = sample.shape[1]
        for chan_idx in range(2):
            for frame_idx in range(seq_len):
                frame = sample[chan_idx][frame_idx].copy()
                frame = cv2.warpAffine(frame, mat, (model.input_width, model.input_height))
                result[chan_idx][frame_idx] = frame

        # Clip or insert some of the frames at head and tail
        head_clip = int(np.round(self.rng.uniform(-self.max_clip_len / 2, self.max_clip_len)))
        tail_clip = int(np.round(self.rng.uniform(-self.max_clip_len / 2, self.max_clip_len)))
        for chan_idx in range(2):
            first_frame, last_frame = result[chan_idx][0], result[chan_idx][-1]
            chan_seq = result[chan_idx].copy()
            if head_clip >= 0:
                chan_seq = chan_seq[head_clip:]
            else:
                head_repeat = np.tile(first_frame, (-head_clip, 1, 1))
                chan_seq = np.append(head_repeat, chan_seq, axis=0)
            if tail_clip >= 0:
                chan_seq = chan_seq[:seq_len - tail_clip]
            else:
                tail_repeat = np.tile(last_frame, (-tail_clip, 1, 1))
                chan_seq = np.append(chan_seq, tail_repeat, axis=0)
            result[chan_idx] = preproc.temporal_resample(chan_seq, seq_len)

        # Apply temporal elastic deformation
        power = 1. / self.rng.uniform(1 - self.temporal_elastic_range, 1 + self.temporal_elastic_range)
        for chan_idx in range(2):
            orig_channel = result[chan_idx].copy()
            for dst_idx in range(seq_len):
                src_idx = np.round(((dst_idx / seq_len) ** power) * seq_len)
                src_idx = int(np.minimum(src_idx, seq_len - 1))
                result[chan_idx][dst_idx] = orig_channel[src_idx]

        # Possibly reverse sequence
        if self.rng.random() < 0.3:
            result = np.flip(result, axis=1)

        # Normalize data channel-wise and return
        return preproc.normalize_sample(result)


if __name__ == '__main__':
    nn = model.CNN3D()
    data_x, data_y = load.from_hdf5("dataset.h5")
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
