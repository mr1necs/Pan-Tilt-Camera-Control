# tests/test_tracker.py

import logging
import cv2

from ulits import parse_args
from modules import VideoCapture, YOLODetector, ObjectTracker


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s: %(message)s'
    )

    args = parse_args()
    device = args['device']
    source = args['camera']

    cap = VideoCapture(source)
    model = YOLODetector(model_path='../models/yolo11n.pt', device=device)
    model.set_class_filter(['frisbee', 'sports ball', 'apple', 'orange', 'cake', 'clock'])

    window_name = 'Traker'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    tracker = None
    frame_count = 0

    while True:
        grabbed, frame = cap.read_frame(width=800)
        if not grabbed or frame is None:
            logging.info("End of stream.")
            break

        frame_count += 1

        if tracker is None or frame_count % 30 == 0:
            boxes = model.detect(frame)
            if boxes:
                x1, y1, x2, y2 = map(int, boxes[0])
                bbox = (x1, y1, x2 - x1, y2 - y1)
                tracker = ObjectTracker('CSRT')
                if not tracker.init(frame, bbox):
                    tracker = None
        else:
            res = tracker.update(frame)
            if res is None:
                tracker = None
            else:
                tx, ty, tw, th = res
                cv2.rectangle(frame, (tx, ty), (tx + tw, ty + th), (0, 255, 0), 2)

        cv2.imshow(window_name, frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            logging.info("Interrupted by user.")
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
