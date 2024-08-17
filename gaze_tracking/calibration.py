from __future__ import division
import cv2
import numpy as np
from .pupil import Pupil


class Calibration(object):
    """
    This class calibrates the pupil detection algorithm by finding
    the best threshold value for the person and the webcam.
    """

    def __init__(self):
        self.nb_frames = 20
        self.thresholds_left = []
        self.thresholds_right = []
        self.screen_points = []
        self.eye_positions = []

    def is_complete(self):
        """Returns true if the calibration is completed"""
        return len(self.thresholds_left) >= self.nb_frames and len(self.thresholds_right) >= self.nb_frames

    def threshold(self, side):
        """Returns the threshold value for the given eye.

        Argument:
            side: 0 for left and 1 for right
        """
        if side == 0:
            return int(sum(self.thresholds_left) / len(self.thresholds_left))
        elif side == 1:
            return int(sum(self.thresholds_right) / len(self.thresholds_right))

    @staticmethod
    def iris_size(frame):
        """Returns the percentage of space that the iris takes up on
        the surface of the eye.

        Argument:
            frame: the iris frame
        """
        frame = frame[5:-5, 5:-5]
        height, width = frame.shape[:2]
        nb_pixels = height * width
        nb_blacks = nb_pixels - cv2.countNonZero(frame)
        return nb_blacks / nb_pixels

    @staticmethod
    def find_best_threshold(eye_frame):
        """Calculates the optimal threshold for the given eye.

        Argument:
            eye_frame: eye's frame to analyse
        """
        average_iris_size = 0.48
        trials = {}

        for threshold in range(5, 100, 5):
            iris_frame = Pupil.image_processing(eye_frame, threshold)
            trials[threshold] = Calibration.iris_size(iris_frame)

        best_threshold, iris_size = min(trials.items(), key=(lambda p: abs(p[1] - average_iris_size)))
        return best_threshold

    def evaluate(self, eye_frame, side):
        """Improves calibration by taking into consideration the
        given image.

        Arguments:
            eye_frame: eye's frame
            side: 0 for left and 1 for right
        """
        threshold = self.find_best_threshold(eye_frame)

        if side == 0:
            self.thresholds_left.append(threshold)
        elif side == 1:
            self.thresholds_right.append(threshold)

    def add_calibration_point(self, eye_position, screen_point):
        """Stores eye positions and corresponding screen points for calibration.

        Arguments:
            eye_position: Tuple (x, y) representing the eye position
            screen_point: Tuple (x, y) representing the screen point
        """
        self.eye_positions.append(eye_position)
        self.screen_points.append(screen_point)

    def get_mapping(self):
        """Returns a polynomial mapping from eye positions to screen coordinates."""
        if len(self.screen_points) < 4:
            return None  # Need at least 4 points to fit a polynomial

        eye_positions_np = np.array(self.eye_positions)
        screen_points_np = np.array(self.screen_points)

        x_coefficients = np.polyfit(eye_positions_np[:, 0], screen_points_np[:, 0], 2)
        y_coefficients = np.polyfit(eye_positions_np[:, 1], screen_points_np[:, 1], 2)

        return x_coefficients, y_coefficients
