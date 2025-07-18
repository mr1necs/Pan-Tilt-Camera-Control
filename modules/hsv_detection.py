# modules/hsv_detection.py

import logging
from typing import List, Tuple, Union

import cv2
import numpy as np


class HSVDetector:
    """
    Wrapper for HSV-based color detection that handles
    HSV range configuration, mask creation, contour detection,
    and drawing bounding boxes.
    """

    def __init__(
            self,
            lower_hsv: Union[Tuple[int, int, int], List[int]],
            upper_hsv: Union[Tuple[int, int, int], List[int]],
            kernel_size: int = 5
    ) -> None:
        """
        Initialize the HSVDetector.

        :param lower_hsv: Lower HSV bound (H, S, V), each 0–255.
        :param upper_hsv: Upper HSV bound (H, S, V), each 0–255.
        :param kernel_size: Size of the morphological kernel (odd integer).
        """
        self.lower_hsv = np.array(lower_hsv, dtype=np.uint8)
        self.upper_hsv = np.array(upper_hsv, dtype=np.uint8)
        # Ensure kernel_size is odd and >= 1
        self.kernel_size = max(1, kernel_size | 1)
        self.kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (self.kernel_size, self.kernel_size)
        )
        logging.info(
            "Initialized HSVDetector with lower=%s, upper=%s, kernel_size=%d",
            tuple(self.lower_hsv), tuple(self.upper_hsv), self.kernel_size
        )

    def set_hsv_range(
            self,
            lower_hsv: Union[Tuple[int, int, int], List[int]],
            upper_hsv: Union[Tuple[int, int, int], List[int]]
    ) -> None:
        """
        Update the HSV detection range.

        :param lower_hsv: New lower HSV bound.
        :param upper_hsv: New upper HSV bound.
        """
        self.lower_hsv = np.array(lower_hsv, dtype=np.uint8)
        self.upper_hsv = np.array(upper_hsv, dtype=np.uint8)
        logging.info(
            "HSV range updated to lower=%s, upper=%s",
            tuple(self.lower_hsv), tuple(self.upper_hsv)
        )

    def set_kernel_size(self, size: int) -> None:
        """
        Update the size of the morphological kernel.

        :param size: New kernel size (odd integer).
        """
        self.kernel_size = max(1, size | 1)
        self.kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (self.kernel_size, self.kernel_size)
        )
        logging.info("Kernel size updated to %d", self.kernel_size)

    def create_mask(self, frame: np.ndarray) -> np.ndarray:
        """
        Convert a BGR frame to HSV and apply the color mask.

        :param frame: Input BGR image.
        :return: Binary mask where detected colors are white.
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        raw_mask = cv2.inRange(hsv, self.lower_hsv, self.upper_hsv)
        # Morphological opening to remove noise
        mask = cv2.morphologyEx(raw_mask, cv2.MORPH_OPEN, self.kernel)
        logging.debug(
            "Mask created with %d non-zero pixels",
            int(cv2.countNonZero(mask))
        )
        return mask

    def detect(self, frame: np.ndarray, min_area: int = 100) -> List[Tuple]:
        """
        Detect contours in the mask and return bounding boxes.

        :param frame: Input BGR image.
        :param min_area: Minimum contour area to keep.
        :return: List of bounding boxes (x, y, w, h).
        """
        mask = self.create_mask(frame)
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        boxes: List[Tuple] = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            boxes.append((x, y, w, h))
        return boxes

    @staticmethod
    def annotate(frame: np.ndarray, boxes: List[Tuple]) -> np.ndarray:
        """
        Draw bounding boxes on the frame.

        :param frame: Original BGR image.
        :param boxes: List of boxes (x, y, w, h).
        :return: Annotated image.
        """
        annotated = frame.copy()
        for (x, y, w, h) in boxes:
            x_i, y_i, w_i, h_i = map(int, (x, y, w, h))
            cv2.rectangle(annotated, (x_i, y_i), (x_i + w_i, y_i + h_i), (0, 255, 0), 2)
        return annotated
