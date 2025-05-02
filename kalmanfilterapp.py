import cv2
import numpy as np
import math

class BallTracker:
    def __init__(self, initial_position):

        # creaciòn del filtro de kalman
        self.kalman = cv2.KalmanFilter(4, 2)
        # 4 variables de estado** (`x, y, vx, vy`) - posición y velocidad.
        self.kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                                                   [0, 1, 0, 0]], np.float32)
        
        self.kalman.transitionMatrix = np.array([[1, 0, 1, 0], # matriz de transición (A)**, que describe cómo cambia el estado de un frame al siguiente.
                                                  [0, 1, 0, 1],
                                                  [0, 0, 1, 0],
                                                  [0, 0, 0, 1]], np.float32)
        
        #Asume que la nueva posición se obtiene sumando la velocidad actual (`x + vx`, `y + vy`).
        #La velocidad se mantiene constante (modelo lineal sin aceleración). 
        self.kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        self.kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.5 # covarianza de la medición (R)
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
