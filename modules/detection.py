import logging
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Tuple, Union

import cv2
import numpy as np
from torch import cuda
from torch.backends import mps
from ultralytics import YOLO


class YOLODetector:
    """
    Wrapper for a YOLO model that handles device selection,
    model initialization, object detection, filtering, and drawing.
    """

    def __init__(self, model_path: Union[str, Path], device: str = 'cpu') -> None:
        """
        Initialize the YOLOModel.

        :param model_path: Path to the YOLO model file.
        :param device: Preferred device ('cpu', 'cuda', 'mps').
        """
        logging.getLogger('ultralytics').setLevel(logging.ERROR)
        self.device = self._choose_device(device)
        self.model = self._initialize_model(str(model_path))
        self.class_filter: Optional[Iterable[str]] = None
        self.confidence_threshold: float = 0.3
        self.iou_threshold: float = 0.45

    def _initialize_model(self, model_path: str) -> YOLO:
        """
        Load the YOLO model and move it to the selected device.

        :param model_path: Path to the model file.
        :return: The initialized YOLO model.
        """
        try:
            model = YOLO(model_path).to(self.device)
            logging.info("YOLO model initialized successfully.")
            return model
        except Exception as e:
            logging.error("Failed to initialize YOLO model: %s", e)
            sys.exit(1)

    @staticmethod
    def _choose_device(device_preference: str) -> str:
        """
        Choose the computation device.

        :param device_preference: Preferred device ('cpu', 'cuda', 'mps').
        :return: Actual device used ('cpu', 'cuda', or 'mps').
        """
        if device_preference == 'mps' and mps.is_available():
            device = 'mps'
        elif device_preference == 'cuda' and cuda.is_available():
            device = 'cuda'
        else:
            if device_preference not in ('cpu', 'cuda', 'mps'):
                logging.warning(
                    "Unknown device '%s'; defaulting to 'cpu'.",
                    device_preference
                )
            elif device_preference != 'cpu':
                logging.warning(
                    "Preferred device '%s' unavailable; using 'cpu'.",
                    device_preference
                )
            device = 'cpu'
        logging.info("Using device: %s", device)
        return device

    def change_device(self, new_device: str) -> None:
        """
        Move the model to a different device.

        :param new_device: Desired device ('cpu', 'cuda', 'mps').
        """
        chosen = self._choose_device(new_device)
        if chosen == self.device:
            logging.info("Device remains unchanged: %s.", self.device)
            return

        self.device = chosen
        self.model.to(self.device)
        logging.info("Model moved to device: %s.", self.device)

    def set_confidence_threshold(self, threshold: float) -> None:
        """
        Set the confidence threshold for detection filtering.

        :param threshold: Value between 0.0 and 1.0.
        """
        self.confidence_threshold = max(0.0, min(1.0, threshold))
        logging.info(
            "Confidence threshold set to %.2f", self.confidence_threshold
        )

    def set_iou_threshold(self, threshold: float) -> None:
        """
        Set the IoU threshold for non-maximum suppression.

        :param threshold: Value between 0.0 and 1.0.
        """
        self.iou_threshold = max(0.0, min(1.0, threshold))
        logging.info("IoU threshold set to %.2f", self.iou_threshold)

    def set_class_filter(self, classes: Optional[Iterable[str]]) -> None:
        """
        Set a filter for detected classes. If None, all classes are allowed.

        :param classes: Iterable of class names or None.
        """
        self.class_filter = set(classes) if classes is not None else None
        logging.info("Class filter set to %s", self.class_filter)

    def detect(self, frame: np.ndarray) -> List[Tuple]:
        """
        Detect objects in a single frame.

        :param frame: BGR image as a numpy.ndarray.
        :return: List of bounding boxes, each as (x1, y1, x2, y2).
        """
        results = self.model(
            frame,
            conf=self.confidence_threshold,
            iou=self.iou_threshold
        )
        boxes: List[Tuple] = []
        for r in results:
            for box in r.boxes:
                score = float(box.conf.cpu().numpy())
                if score < self.confidence_threshold:
                    continue
                cls_id = int(box.cls.cpu().numpy())
                name = self.model.names.get(cls_id, str(cls_id))
                if self.class_filter and name not in self.class_filter:
                    continue
                coords = box.xyxy.cpu().numpy().reshape(-1).tolist()
                boxes.append(tuple(coords))
        return boxes

    @staticmethod
    def annotate(frame: np.ndarray, boxes: List[Tuple]) -> np.ndarray:
        """
        Draw bounding boxes on the frame.

        :param frame: Original image.
        :param boxes: List of bounding boxes from detect().
        :return: Annotated image.
        """
        annotated = frame.copy()
        for x1, y1, x2, y2  in boxes:
            x1_i, y1_i, x2_i, y2_i = map(int, (x1, y1, x2, y2))
            cv2.rectangle(annotated, (x1_i, y1_i), (x2_i, y2_i), (0, 255, 0), 2)
        return annotated

    def get_class_name(self, class_id: int) -> str:
        """
        Get the class name by its index.

        :param class_id: Class index.
        :return: Class name.
        """
        return self.model.names[class_id]


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

