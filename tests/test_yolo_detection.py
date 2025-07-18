import logging
import cv2

from ulits.utils import parse_args
from modules.video_capture import VideoCapture
from modules.detection import YOLODetector


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

    window_name = 'YOLO Detection'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    while True:
        grabbed, frame = cap.read_frame(width=800)
        if not grabbed or frame is None:
            logging.info("End of stream.")
            break

        boxes = model.detect(frame)
        annotated = model.annotate(frame, boxes)
        cv2.imshow(window_name, annotated)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            logging.info("Interrupted by user.")
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
