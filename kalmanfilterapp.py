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
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # rangos amarillo
    lower_yellow = np.array([15, 100, 100])#10 - 50 , 15-43, 
    upper_yellow = np.array([43, 255, 255])

    # rangos naranja
    lower_orange = np.array([7, 130, 130])
    upper_orange = np.array([25, 205, 205])

    # rsngos blancos
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 50, 255])

    # mascaras
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    mask_orange = cv2.inRange(hsv, lower_orange, upper_orange)
    mask_white = cv2.inRange(hsv, lower_white, upper_white)

    # Máscara combinada
    mask = cv2.bitwise_or(mask_yellow, mask_orange)
    mask = cv2.bitwise_or(mask, mask_white)

    mask = cv2.erode(mask, None, iterations=4)
    mask = cv2.dilate(mask, None, iterations=6)

    cv2.imshow('Mascara HSV', mask) # mostrar el filtro de colores

    contornos, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 

    new_balls = []

    for contorno contornos: # analizis de los contrno para determinar si son circulares
        area = cv2.contourArea(contorno)
        if area < 500:
            continue

        perimetro = cv2.arcLength(contorno, True)
        if perimetro == 0:
            continue

        circularidad = 4 * math.pi * area / (perimetro * perimetro)
        if circularidad < 0.7:  # limite de circularidad que buscamos
            continue

        ((x, y), radio) = cv2.minEnclosingCircle(contorno)

        if radio > 10:
            medida = np.array([[np.float32(x)], [np.float32(y)]])
            new_tracker = BallTracker((x, y))
            new_tracker.correct(medida)
            new_balls.append(new_tracker)
            cv2.circle(frame, (int(x), int(y)), int(radio), (0, 255, 0), 2) # Mostramos en la pantalla el contorno del circulo detectado

    balls = new_balls

    for ball in balls:
        prediction = ball.predict()
        predx, predy = int(prediction[0]), int(prediction[1])
        cv2.circle(frame, (predx, predy), 10, (0, 0, 255), 2) # nuevo circlo
        cv2.putText(frame, 'Prediccion', (predx - 30, predy - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2) # marcamos la bola que indica la prediccopn

    cv2.imshow('Seguimiento de Pelotas con Kalman', frame)

cap.release()
cv2.destroyAllWindows()
