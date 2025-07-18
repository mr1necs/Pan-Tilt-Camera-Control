from argparse import ArgumentParser
from typing import Dict, Union


def parse_args() -> Dict[str, Union[str, int, None]]:
    """
    Parse command-line arguments for the video capture and detection pipeline.

    :return: Dictionary of parsed arguments with keys:
             - device: computation device ('mps', 'cuda', or 'cpu')
             - lower_hsv: optional lower HSV bound [H, S, V]
             - upper_hsv: optional upper HSV bound [H, S, V]
             - camera: path to a video file or None to use the default camera
             - buffer: maximum buffer size for trajectory
             - timeout: number of frames before switching to full-image search
             - overlay: size of the ROI overlay
    """
    parser = ArgumentParser(
        description="Capture and detect video frames using YOLO or HSV methods."
    )
    parser.add_argument(
        '-d',
        '--device',
        type=str,
        choices=('mps', 'cuda', 'cpu'),
        default='mps',
        help="Computation device: 'mps', 'cuda', or 'cpu'."
    )
    parser.add_argument(
        '-lh', '--lower-hsv',
        type=int,
        nargs=3,
        metavar=('H', 'S', 'V'),
        default=[29, 86, 6],
        help="Lower HSV bound as three integers: H S V (0–255)."
    )
    parser.add_argument(
        '-uh', '--upper-hsv',
        type=int,
        nargs=3,
        metavar=('H', 'S', 'V'),
        default=[64, 255, 255],
        help="Upper HSV bound as three integers: H S V (0–255)."
    )
    parser.add_argument(
        '-c',
        '--camera',
        type=str,
        default=None,
        help="Path to an optional video file; if not provided, uses default camera."
    )
    parser.add_argument(
        '-b',
        '--buffer',
        type=int,
        default=64,
        help="Maximum buffer size for trajectory."
    )
    parser.add_argument(
        '-t',
        '--timeout',
        type=int,
        default=30,
        help="Number of frames before switching to full-image search."
    )
    parser.add_argument(
        '-o',
        '--overlay',
        type=int,
        default=50,
        help="Size of the ROI overlay (in pixels)."
    )

    args = parser.parse_args()
    return vars(args)
