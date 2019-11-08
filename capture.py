import os
import time
from collections import deque
from threading import Thread
from typing import List, Optional, Callable

import cv2
import numpy as np
import pyrealsense2 as rs

import gesture
import preproc

# Capture
image_width = 640
image_height = 480
record_fps = 16
start_depth_diff = 5  # mean difference of test dequeue indicating beginning of record
finish_depth_diff = 4.5  # mean difference of test dequeue indicating end of record
min_seq_length = 14
test_deque_size = 5  # size of frames temporarily stored in test dequeue


class Camera:
    """
    Simple wrapper of Intel RealSense camera in image capture.
    """

    def __init__(self):
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, image_width, image_height, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, image_width, image_height, rs.format.bgr8, 30)
        profile = self.pipeline.start(config)
        # Get depth sensor's scale
        depth_sensor = profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()
        # Create align object
        self.align = rs.align(rs.stream.color)

    def __del__(self):
        self.pipeline.stop()

    def capture(self) -> List[np.ndarray]:
        """
        Capture an coherent pair of depth and color image.
        :return: list[depth_image, color_image]
        """
        # Wait for a coherent pair of frames
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        if not depth_frame or not color_frame:
            return []

        # Convert images to numpy array
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        return [depth_image, color_image]


class Recorder:
    """
    A video recorder that can record frame sequences for dataset creation and realtime capturing.
    """

    def __init__(self, path: str = None,
                 callback: Callable[[List[np.ndarray]], None] = lambda seq: ()):
        """
        Constructor
        :param path: where to put dataset files
        :param callback: callback function whenever a new valid sequence is recorded
        """
        # Set basic members
        self.gesture = 0  # which gesture to record
        self.camera = Camera()
        if path:
            self.callback = lambda seq: StoreThread(path, self.gesture, seq).start()
            self.train_data = True
        else:  # path not provided, take custom callback
            self.callback = callback
            self.train_data = False
        self.is_recording = False

        # Create recorder GUI
        self.window_name = "Recorder"
        cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)

        def on_track_bar(x: int):
            self.gesture = x

        if self.train_data:
            cv2.createTrackbar("Gesture", self.window_name, self.gesture,
                               len(gesture.category_names) - 1, on_track_bar)

        # Initialize frame buffer
        self.frame_detect = deque(maxlen=test_deque_size)  # store history frames
        self.depth_diff = deque(maxlen=test_deque_size)  # store history difference values

    def record(self) -> None:
        """
        Record a video sequence as train or predict dataset.
        """
        enabled = False  # ready to detect new sequence or not
        delay = Delay(1. / record_fps)
        delay.start()
        while cv2.waitKey(1) != 27:
            # Frame delay
            delay.join()
            delay = Delay(1. / record_fps)
            delay.start()

            # Capture frame from camera
            captured_frame = self.camera.capture()
            if len(captured_frame) == 0:
                continue

            # Preprocess one frame and display it on window
            seg_frame = preproc.segment_one_frame(captured_frame, self.camera.depth_scale)
            self._display(seg_frame)

            # Switch recording state if possible
            if cv2.waitKey(1) == 32:
                print("disabled" if enabled else "enabled")
                enabled = not enabled
                if not enabled:
                    self.is_recording = False
                    self._clear()
            if not enabled:
                continue

            # Try to record a frame
            sequence = self._try_record_frame(seg_frame)
            if sequence is None:
                continue
            self.callback(sequence)  # run callback since a new valid sequence is found

    def _try_record_frame(self, frame: List[np.ndarray]) -> Optional[List[np.ndarray]]:
        """
        Use mean of mask difference to test whether to start or end recording.
        :param frame: a single frame to be recorded
        :return: [depth_sequence, gradient_sequence] if a valid sequence is found
        """
        # Get two images in a frame
        depth_img, grad_img = frame

        # Pop elements if deque is already full
        if len(self.frame_detect) >= test_deque_size:
            self.frame_detect.popleft()
            self.depth_diff.popleft()

        # Push current depth image to deque
        depth_diff, grad_diff = 0, 0
        if len(self.frame_detect) > 0:
            last_depth = self.frame_detect[-1][0]  # get last depth image
            depth_diff = np.mean(np.abs(depth_img - last_depth))
            last_grad = self.frame_detect[-1][1]
        self.frame_detect.append(frame)
        self.depth_diff.append(depth_diff)

        # Compute average difference in recent frames
        depth_diff_mean = np.mean(self.depth_diff)
        # print(depth_diff_mean)
        can_start = depth_diff_mean > start_depth_diff
        if can_start and not self.is_recording:
            self._start_record()
        if self.is_recording:
            should_finish = depth_diff_mean < finish_depth_diff
            if should_finish:
                sequence = self._finish_record()
                return sequence if self._validate_sequence(sequence) else None
            else:
                self.depth_store_list.append(depth_img)
                self.gradient_store_list.append(grad_img)

        return None

    def _start_record(self) -> None:
        """
        Start one round of recording
        :return: None
        """
        self.is_recording = True
        # print("start recording")
        self.depth_store_list = [frame[0] for frame in self.frame_detect]
        self.gradient_store_list = [frame[1] for frame in self.frame_detect]

    def _finish_record(self) -> List[np.ndarray]:
        """
        End this round of recording and return recorded frame sequences.
        :return: [depth_sequence, gradient_sequence]
        """
        self.is_recording = False
        # print("finish recording")
        return [np.array(self.depth_store_list), np.array(self.gradient_store_list)]

    @staticmethod
    def _validate_sequence(seq: List[np.ndarray]) -> bool:
        print("length:", seq[0].shape[0])
        if seq[0].shape[0] < min_seq_length:
            return False
        # print("valid")
        return True

    def _display(self, frame: List[np.ndarray]) -> None:
        """
        Display a single frame of recorded result
        :param frame: [depth_image, gradient_image]
        :return: None
        """
        # Scale image to a suitable size
        window_scale = 0.7
        scaled_width = int(image_width * window_scale)
        scaled_height = int(image_height * window_scale)
        stacked = cv2.resize(np.hstack(frame), (2 * scaled_width, scaled_height))

        # Draw dividing lines and center circles
        cv2.line(stacked, (scaled_width, 0), (scaled_width, scaled_height), 255)
        cv2.circle(stacked, (int(scaled_width * .5), int(scaled_height * .5)), 2, 255,
                   thickness=-1)
        cv2.circle(stacked, (int(scaled_width * 1.5), int(scaled_height * .5)), 2, 255,
                   thickness=-1)

        # Put gesture category text
        if self.train_data:
            cv2.putText(stacked, gesture.category_names[self.gesture], (0, scaled_height),
                        cv2.FONT_HERSHEY_PLAIN, 1, 255)
        cv2.imshow(self.window_name, stacked)

    def _clear(self) -> None:
        """
        Clear all data in buffers.
        :return: None
        """
        self.frame_detect.clear()
        self.depth_diff.clear()
        self.depth_store_list.clear()
        self.gradient_store_list.clear()


class StoreThread(Thread):
    """
    Thread object that managed image sequence storage
    """

    def __init__(self, path: str, gesture_id: int, seq: List[np.ndarray]):
        """
        Constructor
        :param path: where to store image sequence
        :param seq: [depth_sequence, gradient_sequence]
        """
        assert seq
        super().__init__()
        self.path = path
        self.gesture = gesture_id
        self.seq = seq

    def run(self) -> None:
        # Create gesture directory if not found
        gesture_dir = os.path.join(self.path, gesture.category_names[self.gesture])
        if not os.path.exists(gesture_dir):
            os.mkdir(gesture_dir)

        # Create current sequence folder with local time
        seq_dir = os.path.join(gesture_dir, "%d" % int(time.time()))
        os.mkdir(seq_dir)
        print("storing to %s" % seq_dir)

        # Store depth and gradient image, separately
        depth_seq, gradient_seq = self.seq
        for i in range(len(depth_seq)):
            filename = os.path.join(seq_dir, "d%02d.jpg" % i)
            cv2.imwrite(filename, depth_seq[i])
        for i in range(len(gradient_seq)):
            filename = os.path.join(seq_dir, "g%02d.jpg" % i)
            cv2.imwrite(filename, gradient_seq[i])


class Delay(Thread):
    """
    Delay a fixed length of time.
    """

    def __init__(self, delay_len: float):
        """
        Constructor
        :param delay_len: time to be delayed, in seconds
        """
        super().__init__()
        self.len = delay_len

    def run(self) -> None:
        time.sleep(self.len)


if __name__ == '__main__':
    rec = Recorder(path="data")
    # rec = Recorder()
    rec.record()
