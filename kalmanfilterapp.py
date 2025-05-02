import cv2
import numpy as np
import math

class BallTracker:
    def __init__(self, initial_position):
        self.kalman = cv2.KalmanFilter(4, 2) 3 
         # 4 variables de estado** (`x, y, vx, vy`) → posición y velocidad.
        #2 variables de medición** (`x, y`)
        self.kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                                                   [0, 1, 0, 0]], np.float32)

        # matriz de transición (A)
        self.kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                                  [0, 1, 0, 1],
                                                  [0, 0, 1, 0],
                                                  [0, 0, 0, 1]], np.float32)

        
        self.kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        self.kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.5 # covarianza de la medición (R) 
        # representa el ruido o error de nuestras mediciones (`x, y` reales de la cámara).  
        # Un valor de 0.5 significa que confiamos moderadamente en la medición.

        
        self.kalman.statePre = np.array([[initial_position[0]],  # Se establece el estado inicial estimado (x, y, vx, vy) antes de hacer la primera predicción.
                                         [initial_position[1]],
                                         [0],
                                         [0]], np.float32)

    def predict(self): # funciòn para predecir siguiente movimiento usando el filtro de kalman
        return self.kalman.predict()

    def correct(self, measurement):
        self.kalman.correct(measurement)


balls = [] # arreglo para porder detectar mas de un objecto a la vez

cap = cv2.VideoCapture("final.mp4") # video a analizar

while True:


cap.release()
cv2.destroyAllWindows()
