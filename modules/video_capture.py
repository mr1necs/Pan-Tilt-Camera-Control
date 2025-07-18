import logging
import sys
from pathlib import Path
from typing import Optional, Tuple, Union

import cv2
import numpy as np


class VideoCapture:
    """
    Manage a video stream from a file or camera, providing frame
    capture (with optional resizing) and resource cleanup.
    """

    def __init__(self, source: Optional[Union[str, Path, int]] = None ) -> None:
        """
        Initialize the video capture.

        :param source: Path to video file, camera index, or None
            (uses default camera 0).
        """
        self.source = source
        self.cap = self._open_source(source)

    @staticmethod
    def _open_source(source: Optional[Union[str, Path, int]]) -> cv2.VideoCapture:
        """
        Open the video source.

        :param source: Path, camera index, or None.
        :return: cv2.VideoCapture object.
        """
        idx = 0 if source is None else source
        cap = cv2.VideoCapture(idx)
        if not cap.isOpened():
            logging.error("Failed to open video source: %s", idx)
            sys.exit(1)
        logging.info("Video source opened: %s", idx)
        return cap

    @staticmethod
    def resize_frame(
            frame: np.ndarray,
            width: Optional[int] = None
    ) -> np.ndarray:
        """
        Resize a frame to the given width or height, keeping aspect ratio.

        :param frame: Original BGR image.
        :param width: Desired width in pixels; height is computed automatically.
        :return: Resized image.
        """
        h, w = frame.shape[:2]

        if width is None:
            return frame

        new_w = width
        new_h = int(h * width / w)
        resized = cv2.resize(frame, (new_w, new_h))
        return resized

    def read_frame(self, width: Optional[int] = None) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read the next frame from the video stream.

        :param width: If provided, resize frame to this width
            (height is scaled to maintain aspect ratio).
        :return: Tuple (success, frame); frame is None on failure.
        """
        success, frame = self.cap.read()
        if not success:
            logging.info("No more frames or failed to read frame.")
            return False, None

        if width is not None:
            frame = self.resize_frame(frame, width=width)

        return True, frame

    def release(self) -> None:
        """
        Release the video stream resource.
        """
        if self.cap.isOpened():
            self.cap.release()
            logging.info("Video source released.")
