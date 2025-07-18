# tests/test_motors.py
"""
Standalone script to test pan-tilt motors with MotorController.move():
- Pan: 360° clockwise, Tilt: 90° clockwise, then tilt return
- Reverse: Pan 360° counterclockwise, Tilt 90° counterclockwise, then tilt return
"""
import time
import logging
import sys

from modules.motor_controller import MotorController


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger("motor_test")

    # GPIO pin sets: [dir_pin, step_pin, en_pin]
    pan_pins = [23, 24, 25]
    tilt_pins = [17, 27, 22]
    microstep_pins = (21, 21, 21)

    steps_per_rev = 1600
    half_rev = steps_per_rev // 2
    quarter_rev = steps_per_rev // 4

    controller = MotorController(
        pan_pins=pan_pins,
        tilt_pins=tilt_pins,
        microstep_pins=microstep_pins,
        step_type="1/8",
        step_delay=0.0001,
        init_delay=0.05,
    )

    try:
        controller.enable_motors()

        logger.info("Forward: pan to 180°, tilt to 90°")
        controller.move(
            pan_clockwise=True,
            tilt_clockwise=True,
            pan_steps=half_rev,
            tilt_steps=quarter_rev,
        )

        time.sleep(1)

        logger.info("Forward: pan to 360°, tilt to 0°")
        controller.move(
            pan_clockwise=True,
            tilt_clockwise=False,
            pan_steps=half_rev,
            tilt_steps=quarter_rev,
        )

        time.sleep(1)

        logger.info("Forward: pan to -180°, tilt to -90°")
        controller.move(
            pan_clockwise=False,
            tilt_clockwise=False,
            pan_steps=half_rev,
            tilt_steps=quarter_rev,
        )

        time.sleep(1)

        logger.info("Forward pan to -360°, Return tilt to 0°")
        controller.move(
            pan_clockwise=False,
            tilt_clockwise=True,
            pan_steps=half_rev,
            tilt_steps=quarter_rev,
        )

        logger.info("Test sequence completed")

    except Exception:
        logger.exception("Error during motor test")
        sys.exit(1)
    finally:
        controller.disable_motors()
        controller.cleanup()
        logger.info("Cleanup complete")


if __name__ == "__main__":
    main()