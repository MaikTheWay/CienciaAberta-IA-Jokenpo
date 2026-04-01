# DOC;
# Recebe os landmarks da mão e decide se é pedra, papel ou tesoura.
# Abordagem inicial: regras heurísticas com base na distância entre dedos.
# Abordagem avançada: pequeno modelo de aprendizado (ex: MLP) treinado com dados coletados.

from typing import List, Tuple, Dict, Optional


class BrainJokenpo:
    """
    Classificador de pedra/papel/tesoura por regras geométricas.
    Não depende de treino.
    """

    LABELS = ["PEDRA", "PAPEL", "TESOURA"]

    WRIST = 0
    THUMB_CMC = 1
    THUMB_MCP = 2
    THUMB_IP = 3
    THUMB_TIP = 4

    INDEX_MCP = 5
    INDEX_PIP = 6
    INDEX_DIP = 7
    INDEX_TIP = 8

    MIDDLE_MCP = 9
    MIDDLE_PIP = 10
    MIDDLE_DIP = 11
    MIDDLE_TIP = 12

    RING_MCP = 13
    RING_PIP = 14
    RING_DIP = 15
    RING_TIP = 16

    PINKY_MCP = 17
    PINKY_PIP = 18
    PINKY_DIP = 19
    PINKY_TIP = 20

    def __init__(self):
        pass

    def _finger_extended_y(self, landmarks: List[Tuple[float, float, float]], tip_idx: int, pip_idx: int) -> bool:
        if len(landmarks) < 21:
            return False
        return landmarks[tip_idx][1] < landmarks[pip_idx][1]

    def _thumb_extended(self, landmarks: List[Tuple[float, float, float]], handedness: Optional[str]) -> bool:
        """
        Thumb é tratado separadamente porque o eixo muda conforme Left/Right.
        Se handedness estiver indisponível, usamos uma heurística simples.
        """
        if len(landmarks) < 21:
            return False

        thumb_tip_x = landmarks[self.THUMB_TIP][0]
        thumb_ip_x = landmarks[self.THUMB_IP][0]
        index_mcp_x = landmarks[self.INDEX_MCP][0]

        if handedness == "Right":
            return thumb_tip_x < thumb_ip_x
        elif handedness == "Left":
            return thumb_tip_x > thumb_ip_x

        # fallback
        return abs(thumb_tip_x - index_mcp_x) > 0.08

    def classify(self, landmarks: List[Tuple[float, float, float]], handedness: Optional[str] = None) -> Tuple[int, float, Dict[str, bool]]:
        if not landmarks or len(landmarks) < 21:
            return -1, 0.0, {}

        thumb_ext = self._thumb_extended(landmarks, handedness)
        index_ext = self._finger_extended_y(landmarks, self.INDEX_TIP, self.INDEX_PIP)
        middle_ext = self._finger_extended_y(landmarks, self.MIDDLE_TIP, self.MIDDLE_PIP)
        ring_ext = self._finger_extended_y(landmarks, self.RING_TIP, self.RING_PIP)
        pinky_ext = self._finger_extended_y(landmarks, self.PINKY_TIP, self.PINKY_PIP)

        extended_count = sum([thumb_ext, index_ext, middle_ext, ring_ext, pinky_ext])

        debug = {
            "thumb_ext": thumb_ext,
            "index_ext": index_ext,
            "middle_ext": middle_ext,
            "ring_ext": ring_ext,
            "pinky_ext": pinky_ext,
            "extended_count": extended_count,
        }

        # Pedra
        if extended_count <= 1:
            return 0, 0.97, debug

        # Papel
        if extended_count >= 4:
            return 1, 0.97, debug

        # Tesoura
        if index_ext and middle_ext and not ring_ext and not pinky_ext:
            return 2, 0.96, debug

        # Casos intermediários
        if index_ext and middle_ext:
            return 2, 0.72, debug

        if extended_count == 2:
            return 2, 0.65, debug

        if extended_count == 3:
            return 1, 0.60, debug

        return -1, 0.30, debug

    @staticmethod
    def label_to_text(classe: int) -> str:
        if classe == 0:
            return "PEDRA"
        if classe == 1:
            return "PAPEL"
        if classe == 2:
            return "TESOURA"
        return "INDEFINIDO"

    @staticmethod
    def counter_move(jogador_move: str) -> str:
        if jogador_move == "PEDRA":
            return "PAPEL"
        if jogador_move == "PAPEL":
            return "TESOURA"
        if jogador_move == "TESOURA":
            return "PEDRA"
        return "INDEFINIDO"