# modules/motor_controller.py
import logging
import threading
from typing import Sequence, Tuple

import RPi.GPIO as GPIO
from RpiMotorLib import RpiMotorLib


def logging_setup() -> None:
    """Configure the root logger to output at INFO level."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )


class MotorController:
    """
    Control pan and tilt stepper motors using A4988 drivers.
    """

    def __init__(
        self,
        pan_pins: Sequence[int],
        tilt_pins: Sequence[int],
        microstep_pins: Tuple[int, int, int] = (21, 21, 21),
        step_type: str = "1/8",
        step_delay: float = 0.0001,
        init_delay: float = 0.05,
    ) -> None:
        """
        Initialize MotorController with GPIO and motor parameters.

        :param pan_pins: Sequence of three ints: [dir_pin, step_pin, enable_pin] for pan motor.
        :param tilt_pins: Sequence of three ints: [dir_pin, step_pin, enable_pin] for tilt motor.
        :param microstep_pins: Tuple of three GPIO pins for microstep configuration.
        :param step_type: Microstepping type (e.g., "1/8").
        :param step_delay: Delay between steps in seconds.
        :param init_delay: Initial delay before stepping starts.
        """
        logging_setup()

        # Unpack pin configurations into attributes
        self.pan_dir_pin, self.pan_step_pin, self.pan_en_pin = pan_pins
        self.tilt_dir_pin, self.tilt_step_pin, self.tilt_en_pin = tilt_pins

        self.step_type = step_type
        self.step_delay = step_delay
        self.init_delay = init_delay

        # Initialize motor driver instances
        self.pan_motor = RpiMotorLib.A4988Nema(
            self.pan_dir_pin, self.pan_step_pin, microstep_pins, "DRV8825"
        )
        self.tilt_motor = RpiMotorLib.A4988Nema(
            self.tilt_dir_pin, self.tilt_step_pin, microstep_pins, "DRV8825"
        )

        # Configure GPIO for enable pins
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self.pan_en_pin, GPIO.OUT)
        GPIO.setup(self.tilt_en_pin, GPIO.OUT)
        logging.info(
            "GPIO configured: pan_en=%s, tilt_en=%s",
            self.pan_en_pin, self.tilt_en_pin
        )

    def enable_motors(self) -> None:
        """Enable both stepper motors by setting their enable pins low."""
        GPIO.output(self.pan_en_pin, GPIO.LOW)
        GPIO.output(self.tilt_en_pin, GPIO.LOW)
        logging.info("Motors enabled")

    def disable_motors(self) -> None:
        """Disable both stepper motors by setting their enable pins high."""
        GPIO.output(self.pan_en_pin, GPIO.HIGH)
        GPIO.output(self.tilt_en_pin, GPIO.HIGH)
        logging.info("Motors disabled")

    def move(
        self,
        pan_clockwise: bool,
        tilt_clockwise: bool,
        pan_steps: int,
        tilt_steps: int,
    ) -> None:
        """
        Move pan and tilt motors concurrently with independent directions and steps.

        :param pan_clockwise: Direction for pan motor; True for clockwise.
        :param tilt_clockwise: Direction for tilt motor; True for clockwise.
        :param pan_steps: Number of steps for pan motor.
        :param tilt_steps: Number of steps for tilt motor.
        """
        threads = []

        # Create thread for pan motor
        pan_thread = threading.Thread(
            target=self._move_motor,
            args=(self.pan_motor, pan_clockwise, pan_steps),
            name="pan_move"
        )
        threads.append(pan_thread)
        pan_thread.start()
        logging.info(
            "Started pan thread %s: %d steps %s",
            pan_thread.name, pan_steps, 'CW' if pan_clockwise else 'CCW'
        )

        # Create thread for tilt motor
        tilt_thread = threading.Thread(
            target=self._move_motor,
            args=(self.tilt_motor, tilt_clockwise, tilt_steps),
            name="tilt_move"
        )
        threads.append(tilt_thread)
        tilt_thread.start()
        logging.info(
            "Started tilt thread %s: %d steps %s",
            tilt_thread.name, tilt_steps, 'CW' if tilt_clockwise else 'CCW'
        )

        # Wait for both to finish
        for thread in threads:
            thread.join()
            logging.info("Thread %s finished", thread.name)

    def _move_motor(
        self,
        motor: RpiMotorLib.A4988Nema,
        clockwise: bool,
        steps: int,
    ) -> None:
        """
        Move a single motor with given direction and number of steps.

        :param motor: Instance of A4988Nema motor to control.
        :param clockwise: True for clockwise movement, False otherwise.
        :param steps: Number of steps to execute.
        """
        motor.motor_go(
            clockwise=clockwise,
            steptype=self.step_type,
            steps=steps,
            stepdelay=self.step_delay,
            verbose=False,
            initdelay=self.init_delay,
        )
        direction = 'clockwise' if clockwise else 'counterclockwise'
        logging.info(
            "Motor %s moved %d steps %s",
            motor, steps, direction
        )

    def cleanup(self) -> None:
        """Cleanup GPIO resources used by the motors."""
        GPIO.cleanup()
        logging.info("GPIO cleaned up")
