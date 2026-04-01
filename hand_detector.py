# Usa MediaPipe Hands para localizar pontos de referência da mão no frame da câmera.
# Retorna True quando uma mão é detectada e está em posição adequada (ex: aberta, visível).

import cv2
import mediapipe as mp
from dataclasses import dataclass
from typing import List, Optional, Tuple, Any

@dataclass
class HandDetectionResult:
    visible: bool
    landmarks: List[Tuple[float, float, float]]
    flat_landmarks: List[float]
    handedness: Optional[str]
    annotated_frame: Any


class HandDetector:
    """
    Localiza landmarks da mão usando MediaPipe Hands.
    """
    def __init__(
        self,
        max_num_hands: int = 1,
        detection_confidence: float = 0.7,
        tracking_confidence: float = 0.7,
    ):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
        )
        self.mp_draw = mp.solutions.drawing_utils

    def encontrar_pontos(self, frame) -> HandDetectionResult:
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(img_rgb)

        landmarks: List[Tuple[float, float, float]] = []
        flat_landmarks: List[float] = []
        handedness: Optional[str] = None
        visible = False

        if results.multi_hand_landmarks:
            visible = True

            if results.multi_handedness and len(results.multi_handedness) > 0:
                handedness = results.multi_handedness[0].classification[0].label

            hand_lms = results.multi_hand_landmarks[0]
            for lm in hand_lms.landmark:
                x, y, z = lm.x, lm.y, lm.z
                landmarks.append((x, y, z))
                flat_landmarks.extend([x, y, z])

            self.mp_draw.draw_landmarks(
                frame,
                hand_lms,
                self.mp_hands.HAND_CONNECTIONS,
            )

        return HandDetectionResult(
            visible=visible,
            landmarks=landmarks,
            flat_landmarks=flat_landmarks,
            handedness=handedness,
            annotated_frame=frame,
        )