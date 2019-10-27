import numpy as np
import pyrealsense2 as rs

# Capture
image_width = 640
image_height = 480
fps = 30


class Camera:
    """
    Intel RealSense camera for use in image capture.
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

    def capture(self) -> (np.ndarray, np.ndarray):
        """
        Capture an coherent pair of depth and color image.
        :return: tuple(depth_img, color_img)
        """
        # Wait for a coherent pair of frames
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        if not depth_frame or not color_frame:
            return None, None

        # Convert images to np.array
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        return depth_image, color_image
