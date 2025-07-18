import logging
import cv2

from ulits.utils import parse_args
from modules.video_capture import VideoCapture
from modules.detection import HSVDetector


def main() -> None:

    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s: %(message)s'
    )

    args = parse_args()
    source = args['camera']


    cap = VideoCapture(source)

    model = HSVDetector(lower_hsv = (29, 86, 6), upper_hsv = (64, 255, 255))

    window_name = 'HSV Detection'
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
