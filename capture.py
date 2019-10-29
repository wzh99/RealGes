from collections import deque
from typing import List, Union

import cv2
import numpy as np
import pyrealsense2 as rs

import gesture
import preproc

# Capture
image_width = 640
image_height = 480
fps = 30
start_diff_threshold = 10  # mean of test dequeue indicating when the recording should begin
finish_diff_threshold = 10  # mean of test dequeue indicating when the recording should end
min_seq_length = 10
test_deque_size = 4  # size of frames temporarily stored in test dequeue


class Camera:
    """
    Simple wrapper of Intel RealSense camera in image capture.
    """

    def __init__(self):
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, image_width, image_height, rs.format.z16, fps)
        config.enable_stream(rs.stream.color, image_width, image_height, rs.format.bgr8, fps)
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

    def __init__(self, camera: Camera, path: str = "./"):
        """
        Constructor
        :param camera: Camera object
        :param path: where to put dataset files
        """
        # Set basic members
        self.gesture = 0  # which gesture to record
        self.camera = camera
        self.path = path
        self.is_recording = False

        # Create recorder GUI
        self.window_name = "Recorder"
        cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)

        def on_track_bar(x: int):
            self.gesture = x

        cv2.createTrackbar("Gesture", self.window_name, self.gesture, len(gesture.category_names) - 1,
                           on_track_bar)

        # Initialize frame buffer
        self.frame_test_deque = deque(maxlen=test_deque_size)  # store history frames
        self.diff_test_deque = deque(maxlen=test_deque_size)  # store history mean values

    def record(self) -> Union[np.ndarray, None]:
        """
        Record a video sequence as train dataset and store it into directory.
        :return: numpy array if recording real time test data, or None if creating dataset.
        """
        enabled = False  # ready to detect new sequence or not
        while cv2.waitKey(1) != 27:
            captured_frame = self.camera.capture()
            if len(captured_frame) == 0:
                continue
            seg_frame = preproc.segment_one_frame(captured_frame, self.camera.depth_scale)
            self._display(seg_frame)
            if cv2.waitKey(1) == 32:
                print("disabled" if enabled else "enabled")
                enabled = not enabled
            if enabled:
                self._try_record_frame(seg_frame)

        return None

    def _try_record_frame(self, frame: List[np.ndarray]) -> Union[List[np.ndarray], None]:
        """
        Use mean of mask difference to test whether to start or end recording.
        :param frame: a single frame to be recorded
        :return: None
        """
        # Get two images in a frame
        depth_image, gradient_image = frame

        # Pop elements if deque is already full
        if len(self.frame_test_deque) >= test_deque_size:
            self.frame_test_deque.popleft()
            self.diff_test_deque.popleft()

        # Push current depth image to deque
        this_diff = 0
        if len(self.frame_test_deque) > 0:
            last_depth_image = self.frame_test_deque[-1][0]  # get last depth image
            diff_image = depth_image - last_depth_image
            this_diff = cv2.mean(cv2.inRange(diff_image, 10, 255))[0]
            # print(this_diff)
        self.frame_test_deque.append(frame)
        self.diff_test_deque.append(this_diff)

        # Compute average difference in recent frames
        mean = np.mean(self.diff_test_deque)
        if mean > start_diff_threshold and not self.is_recording:
            self._start_record()
        if self.is_recording:
            if mean < finish_diff_threshold:
                sequence = self._finish_record()
                return sequence if self._check_sequence(sequence) else None
            else:
                self.depth_store_list.append(depth_image)
                self.gradient_store_list.append(gradient_image)

        return None

    def _start_record(self):
        """
        Start one round of recording
        :return:
        """
        self.is_recording = True
        # print("Start recording")
        self.depth_store_list = [frame[0] for frame in self.frame_test_deque]
        self.gradient_store_list = [frame[1] for frame in self.frame_test_deque]

    def _finish_record(self) -> List[np.ndarray]:
        """
        End this round of recording and return recorded frame sequences.
        :return: [depth_sequence, gradient_sequence]
        """
        self.is_recording = False
        # print("Finish recording")
        return [np.array(self.depth_store_list), np.array(self.gradient_store_list)]

    @staticmethod
    def _check_sequence(seq: List[np.ndarray]) -> bool:
        print("length:", seq[0].shape[0])
        if seq[0].shape[0] < min_seq_length:
            return False
        print("valid")
        return True

    def _display(self, frame: List[np.ndarray]):
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
        cv2.circle(stacked, (int(scaled_width * .5), int(scaled_height * .5)), 2, 255, thickness=-1)
        cv2.circle(stacked, (int(scaled_width * 1.5), int(scaled_height * .5)), 2, 255, thickness=-1)

        # Put gesture category text
        cv2.putText(stacked, gesture.category_names[self.gesture], (0, scaled_height),
                    cv2.FONT_HERSHEY_PLAIN, 1, 255)
        cv2.imshow(self.window_name, stacked)


if __name__ == '__main__':
    recorder = Recorder(Camera(), path="./data/")
    recorder.record()
