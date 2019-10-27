import cv2
import numpy as np

from capture import Camera

# Preprocess
depth_thresh = 0.4  # in meters
median_size = 7  # median filter kernel size
dilation_size = 5  # dilation kernel size
cr_range = (133, 173)  # Cr component range
cb_range = (77, 127)  # Cb component range
min_contour_area = 10000  # minimum acceptable hand contour area

if __name__ == '__main__':
    camera = Camera()
    while cv2.waitKey(1) != 27:
        # Retrieve image from camera
        depth_image, color_image = camera.capture()

        # Create mask from depth
        # Filter out depth that fall out of accepted range
        depth_mask = cv2.inRange(depth_image, 1, depth_thresh / camera.depth_scale)
        # Use median filter to smooth jagged edge
        depth_mask = cv2.medianBlur(depth_mask, median_size)
        # Expand the mask a bit
        depth_mask = cv2.dilate(depth_mask, np.ones((dilation_size, dilation_size)))

        # Create mask from color
        # De-noise with median filter
        blurred_color = cv2.medianBlur(color_image, median_size)
        # Convert to YUV color space
        ycrcb_image = cv2.cvtColor(blurred_color, cv2.COLOR_BGR2YCrCb)
        # Split into separate channels
        ycrcb_split = cv2.split(ycrcb_image)
        # Create mask according to uv range
        cr_mask = cv2.inRange(ycrcb_split[1], cr_range[0], cr_range[1])
        cb_mask = cv2.inRange(ycrcb_split[2], cb_range[0], cb_range[1])
        color_mask = cr_mask & cb_mask

        # Combine two masks
        combined_mask = depth_mask & color_mask

        # Find contour of the mask
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        hand_contour = None
        if len(contours) > 0:
            hand_contour = max(contours, key=cv2.contourArea)
            # Reject this contour if it's too small
            if cv2.contourArea(hand_contour) < min_contour_area:
                hand_contour = None

        # Draw hand contour as mask if it is found
        hand_mask = np.zeros(combined_mask.shape, dtype=np.uint8)
        if hand_contour is not None:
            poly_points = hand_contour.reshape(len(hand_contour), 2)
            cv2.fillPoly(hand_mask, [poly_points], 255)
        masked_color = cv2.copyTo(color_image, hand_mask)
        cv2.imshow("Segmentation", masked_color)
