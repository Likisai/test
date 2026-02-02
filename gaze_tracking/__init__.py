import cv2
import numpy as np

class GazeTracking:
    def __init__(self):
        self.eye_left = None
        self.eye_right = None

    def refresh(self, frame):
        self.frame = frame

    def is_left(self):
        return False

    def is_right(self):
        return False

    def is_center(self):
        return True

    def is_blinking(self):
        return False

    def annotated_frame(self):
        return self.frame
