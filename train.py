import os
import time
from random import Random
from threading import Thread

import cv2
import keras
from keras import backend as K
from keras.callbacks import ModelCheckpoint, TensorBoard
import numpy as np

import gesture
import load
import model
import preproc


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
        self.rng = Random()
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
        power = 1. / self.rng.uniform(1 - self.temporal_elastic_range,
                                      1 + self.temporal_elastic_range)
        for chan_idx in range(2):
            orig_channel = result[chan_idx].copy()
            for dst_idx in range(seq_len):
                src_idx = np.round(((dst_idx / seq_len) ** power) * seq_len)
                src_idx = int(np.minimum(src_idx, seq_len - 1))
                result[chan_idx][dst_idx] = orig_channel[src_idx]

        # Normalize data channel-wise and return
        return preproc.normalize_sample(result)


class ExitListener(Thread):
    """
    A daemon thread that polls enter key and inform training program to exit.
    """

    def __init__(self):
        super().__init__()
        self.setDaemon(True)
        self.exit = False

    def run(self) -> None:
        while True:
            if not input():
                self.exit = True
                print("Will exit when this epoch is finished.")
                break


class Trainer:
    """
    Trains network with gesture data.
    """

    def __init__(self, spec: dict, data_path: str, patience: int = 10, decay: float = 0.5):
        """
        Constructor
        :param spec: dict object specifying constructor and weight path of a model
        :param data_path: where to load data file (in HDF5 format)
        :param patience: number of epochs that can be waited until a learning rate decay
        :param decay: decaying rate of learning rate
        """
        # Initialize members
        self.spec = spec
        self.model: keras.models.Model = spec["init"]()
        self.data_path = data_path
        self.patience = patience
        assert decay < 1
        self.decay = decay

        # Possibly load weight file if it is found
        if os.path.exists(self.spec["path"]):
            print("Model file is found.")
            self.model.load_weights(spec["path"])

    def train(self, num_epochs: int):
        """
        Perform main training work on the specified model.
        """
        # Load dataset from file and initialize data augmentation
        data_x, data_y = load.from_hdf5(self.data_path)
        data_y = keras.utils.to_categorical(data_y, len(gesture.category_names))
        aug = Augmentor(data_x)
        aug.start()

        # Start exit listener thread
        listener = ExitListener()
        listener.start()

        # Set training callback
        checkpoint = ModelCheckpoint(self.spec["path"], monitor="loss", verbose=1, 
                                     save_best_only=True)

        # Training loop
        epoch_idx = 0
        last_update = 0  # last epoch we got a lower loss or updated learning rate
        lowest_loss = float("inf")

        while not listener.exit and epoch_idx < num_epochs:
            print("Epoch %d/%d" % (epoch_idx, num_epochs))
            aug.join()
            aug_data_x = aug.result.copy()
            aug = Augmentor(data_x)  # a single thread object cannot be started more than once
            aug.start()  # run data augmentation of next epoch concurrently with current training
            history = self.model.fit(x=aug_data_x, y=data_y, batch_size=20, callbacks=[checkpoint])
            cur_loss = history.history["loss"][0]
            if cur_loss < lowest_loss:
                last_update = epoch_idx
                lowest_loss = cur_loss
            elif epoch_idx - last_update > self.patience:
                learning_rate = K.get_value(self.model.optimizer.lr)
                new_rate = learning_rate * self.decay
                print("Learning rate decayed to %f" % (new_rate))
                K.set_value(self.model.optimizer.lr, new_rate)
                last_update = epoch_idx
            epoch_idx += 1


if __name__ == '__main__':
    trainer = Trainer(model.network_spec["hrn"], "dataset.h5")
    trainer.train(100)
