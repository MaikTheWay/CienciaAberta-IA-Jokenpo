# Usa MediaPipe Hands para localizar pontos de referência da mão no frame da câmera.
# Retorna True quando uma mão é detectada e está em posição adequada (ex: aberta, visível).

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

class HandDetector:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
        self.mp_draw = mp.solutions.drawing_utils

    def encontrar_pontos(self, frame):
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(img_rgb)
        lista_pontos = []
        
        if self.results.multi_hand_landmarks:
            for hand_lms in self.results.multi_hand_landmarks:
                # Extrai os 21 pontos (x, y, z) em uma lista flat de 63 valores
                for lm in hand_lms.landmark:
                    lista_pontos.extend([lm.x, lm.y, lm.z])
                self.mp_draw.draw_landmarks(frame, hand_lms, self.mp_hands.HAND_CONNECTIONS)
        return lista_pontos, frame
