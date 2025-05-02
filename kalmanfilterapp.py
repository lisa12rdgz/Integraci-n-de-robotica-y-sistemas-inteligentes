import cv2
import numpy as np
import math

class BallTracker:
    def __init__(self, initial_position):
        self.kalman = cv2.KalmanFilter(4, 2)
        self.kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                                                   [0, 1, 0, 0]], np.float32)
        
        self.kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                                  [0, 1, 0, 1],
                                                  [0, 0, 1, 0],
                                                  [0, 0, 0, 1]], np.float32)
        self.kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        self.kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.5
        self.kalman.statePre = np.array([[initial_position[0]],
                                         [initial_position[1]],
                                         [0],
                                         [0]], np.float32)

    def predict(self):
        return self.kalman.predict()

    def correct(self, measurement):
        self.kalman.correct(measurement)


balls = []

cap = cv2.VideoCapture("final.mp4")

while True:


cap.release()
cv2.destroyAllWindows()
