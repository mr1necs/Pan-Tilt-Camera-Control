# modules/object_tracker.py

import logging
from typing import Optional, Tuple, Callable

import cv2
import numpy as np

_TRACKERS: dict[str, Callable[[], cv2.Tracker]] = {
    'KCF': cv2.legacy.TrackerKCF_create,
    'TLD': cv2.legacy.TrackerTLD_create,
    'MOSSE': cv2.legacy.TrackerMOSSE_create,
    'CSRT': cv2.legacy.TrackerCSRT_create,
}


class ObjectTracker:
    """
    Wrapper for OpenCV object trackers.
    Supported types: KCF, TLD, MOSSE, CSRT.
    """

    def __init__(self, tracker_type: str = 'CSRT') -> None:
        """
        Initialize tracker instance based on tracker_type.

        :param tracker_type: Name of the tracker to use.
        :raises ValueError: If the tracker type is unsupported.
        """
        create_fn = _TRACKERS.get(tracker_type.upper())
        if create_fn is None:
            raise ValueError(f"Unsupported tracker type '{tracker_type.upper()}'")

        self.tracker = create_fn()
        self.initialized = False

    def init(self, frame: np.ndarray, bbox: Tuple) -> bool:
        """
        Initialize tracker with the first frame and bounding box.

        :param frame: BGR image array.
        :param bbox: (x, y, width, height).
        :return: True on success, False otherwise.
        """
        success = self.tracker.init(frame, bbox)
        self.initialized = bool(success)
        if not success:
            logging.error("Failed to initialize tracker with bbox %s", bbox)
        return self.initialized

    def update(self, frame: np.ndarray) -> Optional[Tuple]:
        """
        Update tracker for a new frame.

        :param frame: New BGR frame.
        :return: (x, y, w, h) on success, or None on failure.
        """
        result = self.tracker.update(frame)

        try:
            ok, bbox = result
        except (ValueError, TypeError):
            ok, bbox = True, result

        if not ok:
            return None

        if len(bbox) == 1 and isinstance(bbox[0], tuple):
            bbox, = bbox

        return tuple(map(int, bbox))
