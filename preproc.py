from typing import List

import cv2
import numpy as np

# Preprocess
depth_thresh = 0.4  # in meters
median_size = 7  # median filter kernel size
dilation_size = 5  # dilation kernel size
cr_range = (133, 173)  # Cr component range
cb_range = (77, 127)  # Cb component range
min_contour_area = 10000  # minimum acceptable hand contour area


def segment_one_frame(captured_list: List[np.ndarray], depth_scale: float) -> List[np.ndarray]:
    """
    Segment out hand in one pair of depth and gradient images
    :return: [segmented_depth, segmented_gradient]
    """
    # Split image list
    depth_image, color_image = captured_list[0], captured_list[1]

    # Create mask from depth
    # Filter out depth that fall out of accepted range
    depth_mask = cv2.inRange(depth_image, 1, depth_thresh / depth_scale)
    # Use median filter to smooth jagged edge
    depth_mask = cv2.medianBlur(depth_mask, median_size)
    # Expand the mask a bit
    depth_mask = cv2.dilate(depth_mask, np.ones((dilation_size, dilation_size)))
    # Fill holes in depth mask
    depth_mask = fill_hole(depth_mask)

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
    # Fill holes in color mask
    color_mask = fill_hole(color_mask)

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

    # Show masked depth image
    scaled_depth = cv2.convertScaleAbs(depth_image, alpha=0.03)
    masked_depth = cv2.copyTo(scaled_depth, hand_mask)

    # Compute gradient of grayscale image
    gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
    gradient_image = cv2.Sobel(gray_image, -1, 1, 1, ksize=5)
    masked_gradient = cv2.copyTo(gradient_image, hand_mask)

    return [masked_depth, masked_gradient]


def fill_hole(mask: np.ndarray) -> np.ndarray:
    """
    File holes in the mask
    :param mask: mask to be filled
    :return: filled result
    """
    shape = np.array(mask.shape)
    fill = np.zeros(shape + 2, np.uint8)
    fill[1:shape[0] + 1, 1:shape[1] + 1] = mask.copy()
    cv2.floodFill(fill, np.zeros(shape + 4, np.uint8), (0, 0), 255)
    return mask | ~fill[1:shape[0] + 1, 1:shape[1] + 1]


if __name__ == '__main__':
    from capture import Camera
    camera = Camera()
    while cv2.waitKey(1) != 27:
        segmented = segment_one_frame(camera.capture(), camera.depth_scale)
        cv2.imshow("Result", np.hstack(segmented))
