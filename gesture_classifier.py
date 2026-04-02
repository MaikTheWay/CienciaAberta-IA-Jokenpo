# =============================================================================
# gesture_classifier.py
# =============================================================================
# Classificador de gestos para Pedra-Papel-Tesoura.
# Usa regras geométricas com distâncias euclidianas para detecção robusta.
#
# Melhorias:
# - Distância euclidiana entre pontas dos dedos e palma
# - Comparação relativa entre dedos
# - Verificação de fechamento da mão baseado em distâncias
#
# Alberto Seleto de Souza / Marcos Alcino Ribeiro Cussioli
# =============================================================================

import math
from typing import List, Tuple, Dict, Optional


class BrainJokenpo:
    """
    Classificador de pedra/papel/tesoura por regras geométricas.
    Usa distâncias euclidianas para detecção mais robusta.
    """

    LABELS = ["PEDRA", "PAPEL", "TESOURA"]

    # Índices dos landmarks da mão (MediaPipe)
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

    # Landmarks da palma (para normalização)
    PALM_INDICES = [0, 1, 5, 9, 13, 17]  # Wrist + bases dos dedos

    def __init__(self, distance_threshold: float = 0.08):
        """
        Inicializa o classificador.

        Args:
            distance_threshold: Limiar de distância para considerar dedo estendido
        """
        self.distance_threshold = distance_threshold

    def _euclidean_distance(self, p1: Tuple[float, float, float],
                           p2: Tuple[float, float, float]) -> float:
        """Calcula distância euclidiana 3D entre dois pontos."""
        return math.sqrt(
            (p1[0] - p2[0]) ** 2 +
            (p1[1] - p2[1]) ** 2 +
            (p1[2] - p2[2]) ** 2
        )

    def _get_palm_center(self, landmarks: List[Tuple[float, float, float]]) -> Tuple[float, float, float]:
        """Calcula o centro da palma baseado nos landmarks da base."""
        palm_indices = [self.WRIST, self.INDEX_MCP, self.MIDDLE_MCP, self.RING_MCP, self.PINKY_MCP]
        cx = sum(landmarks[i][0] for i in palm_indices) / len(palm_indices)
        cy = sum(landmarks[i][1] for i in palm_indices) / len(palm_indices)
        cz = sum(landmarks[i][2] for i in palm_indices) / len(palm_indices)
        return (cx, cy, cz)

    def _finger_extended_distance(
        self,
        landmarks: List[Tuple[float, float, float]],
        tip_idx: int,
        pip_idx: int,
        mcp_idx: int
    ) -> Tuple[bool, float]:
        """
        Verifica se um dedo está estendido usando distância euclidiana.

        Returns:
            Tuple (is_extended, distance_ratio)
        """
        if len(landmarks) < 21:
            return False, 0.0

        tip = landmarks[tip_idx]
        pip = landmarks[pip_idx]
        mcp = landmarks[mcp_idx]
        wrist = landmarks[self.WRIST]

        # Distância da ponta até o pulso
        tip_to_wrist = self._euclidean_distance(tip, wrist)

        # Distância do MCP até o pulso (comprimento de referência)
        mcp_to_wrist = self._euclidean_distance(mcp, wrist)

        # Razão - se a ponta está mais longe que o MCP, dedo está estendido
        ratio = tip_to_wrist / (mcp_to_wrist + 0.001)

        # Limiar: razão > 1.3 indica dedo estendido
        is_extended = ratio > 1.3

        return is_extended, ratio

    def _thumb_extended(
        self,
        landmarks: List[Tuple[float, float, float]],
        handedness: Optional[str]
    ) -> Tuple[bool, float]:
        """
        Detecta se o polegar está estendido.
        O polegar é especial porque fica no plano horizontal.
        """
        if len(landmarks) < 21:
            return False, 0.0

        thumb_tip = landmarks[self.THUMB_TIP]
        thumb_ip = landmarks[self.THUMB_IP]
        thumb_mcp = landmarks[self.THUMB_MCP]
        index_mcp = landmarks[self.INDEX_MCP]

        # Distância do polegar até a base do indicador
        thumb_to_index = self._euclidean_distance(thumb_tip, index_mcp)

        # Distância do IP até MCP (referência)
        ip_to_mcp = self._euclidean_distance(thumb_ip, thumb_mcp)

        # Razão
        ratio = thumb_to_index / (ip_to_mcp + 0.001)

        # Polegar estendido se a ponta está longe do indicador
        # Limiar varia conforme a mão (esquerda/direita)
        if handedness == "Right":
            # Polegar aponta para esquerda
            is_extended = thumb_tip[0] < thumb_ip[0] - 0.02
        elif handedness == "Left":
            # Polegar aponta para direita
            is_extended = thumb_tip[0] > thumb_ip[0] + 0.02
        else:
            # Fallback: usa distância
            is_extended = ratio > 1.8

        return is_extended, ratio

    def _is_hand_closed(
        self,
        landmarks: List[Tuple[float, float, float]]
    ) -> bool:
        """
        Verifica se a mão está fechada (para detectar PEDRA).
        Usa distâncias entre pontas dos dedos e a palma.
        """
        if len(landmarks) < 21:
            return False

        palm_center = self._get_palm_center(landmarks)

        # Pontas dos dedos
        tips = [
            landmarks[self.THUMB_TIP],
            landmarks[self.INDEX_TIP],
            landmarks[self.MIDDLE_TIP],
            landmarks[self.RING_TIP],
            landmarks[self.PINKY_TIP]
        ]

        # Calcula distância média das pontas até o centro da palma
        avg_distance = sum(
            self._euclidean_distance(tip, palm_center)
            for tip in tips
        ) / len(tips)

        # Se distância média é pequena, mão está fechada
        return avg_distance < 0.20

    def _is_hand_open(
        self,
        landmarks: List[Tuple[float, float, float]],
        finger_distances: Dict[str, float]
    ) -> bool:
        """
        Verifica se a mão está aberta (para detectar PAPEL).
        """
        # Todos os dedos devem estar estendidos com alta razão
        required_extended = 4  # Pelo menos 4 dedos estendidos
        extended_count = sum(
            1 for ratio in finger_distances.values()
            if ratio > 1.3
        )

        return extended_count >= required_extended

    def _is_scissors(
        self,
        finger_states: Dict[str, bool],
        finger_distances: Dict[str, float]
    ) -> bool:
        """
        Detecta tesoura: indicador e médio estendidos, anelar e mindinho fechados.
        """
        index_ok = finger_states.get('index', False) and finger_distances.get('index', 0) > 1.4
        middle_ok = finger_states.get('middle', False) and finger_distances.get('middle', 0) > 1.4
        ring_closed = not (finger_states.get('ring', True)) or finger_distances.get('ring', 0) < 1.3
        pinky_closed = not (finger_states.get('pinky', True)) or finger_distances.get('pinky', 0) < 1.3

        return index_ok and middle_ok and ring_closed and pinky_closed

    def classify(
        self,
        landmarks: List[Tuple[float, float, float]],
        handedness: Optional[str] = None
    ) -> Tuple[int, float, Dict]:
        """
        Classifica o gesto da mão.

        Returns:
            Tuple (class_id, confidence, debug_info)
            class_id: 0=PEDRA, 1=PAPEL, 2=TESOURA, -1=INDEFINIDO
        """
        if not landmarks or len(landmarks) < 21:
            return -1, 0.0, {}

        # Detecta estado de cada dedo
        thumb_ext, thumb_ratio = self._thumb_extended(landmarks, handedness)
        index_ext, index_ratio = self._finger_extended_distance(
            landmarks, self.INDEX_TIP, self.INDEX_PIP, self.INDEX_MCP
        )
        middle_ext, middle_ratio = self._finger_extended_distance(
            landmarks, self.MIDDLE_TIP, self.MIDDLE_PIP, self.MIDDLE_MCP
        )
        ring_ext, ring_ratio = self._finger_extended_distance(
            landmarks, self.RING_TIP, self.RING_PIP, self.RING_MCP
        )
        pinky_ext, pinky_ratio = self._finger_extended_distance(
            landmarks, self.PINKY_TIP, self.PINKY_PIP, self.PINKY_MCP
        )

        finger_states = {
            'thumb': thumb_ext,
            'index': index_ext,
            'middle': middle_ext,
            'ring': ring_ext,
            'pinky': pinky_ext
        }

        finger_distances = {
            'thumb': thumb_ratio,
            'index': index_ratio,
            'middle': middle_ratio,
            'ring': ring_ratio,
            'pinky': pinky_ratio
        }

        extended_count = sum(finger_states.values())

        debug = {
            'finger_states': finger_states,
            'finger_distances': finger_distances,
            'extended_count': extended_count,
            'hand_closed': self._is_hand_closed(landmarks)
        }

        # Lógica de classificação melhorada

        # 1. TESOURA: Indicador e médio estendidos, anelar e mindinho fechados
        if self._is_scissors(finger_states, finger_distances):
            return 2, 0.95, debug

        # 2. PEDRA: Mão fechada ou poucos dedos estendidos
        # Condição: mão fechada OU 0-1 dedos estendidos
        if self._is_hand_closed(landmarks) or extended_count <= 1:
            return 0, 0.95, debug

        # 3. PAPEL: Todos os dedos estendidos
        # Condição: polegar + todos os 4 dedos estendidos
        if thumb_ext and index_ext and middle_ext and ring_ext and pinky_ext:
            return 1, 0.95, debug

        # 4. Casos intermediários com mais contexto

        # Se tem 4 dedos estendidos (exceto polegar variável)
        if extended_count >= 4 and not thumb_ext:
            # Provavelmente papel com polegar parcialmente fechado
            return 1, 0.85, debug

        if extended_count >= 4 and thumb_ext:
            return 1, 0.92, debug

        # Se tem 2 dedos estendidos (indicador + médio)
        if index_ext and middle_ext and not ring_ext and not pinky_ext:
            return 2, 0.88, debug

        # Se tem 2 dedos estendidos (outros casos)
        if extended_count == 2:
            #可能是 tesoura ou transição
            if index_ext or middle_ext:
                return 2, 0.70, debug
            return 0, 0.60, debug

        # Se tem 3 dedos estendidos
        if extended_count == 3:
            # Mais provável papel com 1 dedo parcialmente fechado
            return 1, 0.65, debug

        # Caso padrão: indefinido
        return -1, 0.30, debug

    @staticmethod
    def label_to_text(classe: int) -> str:
        """Converte ID da classe para texto."""
        if classe == 0:
            return "PEDRA"
        if classe == 1:
            return "PAPEL"
        if classe == 2:
            return "TESOURA"
        return "INDEFINIDO"

    @staticmethod
    def counter_move(jogador_move: str) -> str:
        """Retorna a jogada que contra-ataca."""
        if jogador_move == "PEDRA":
            return "PAPEL"
        if jogador_move == "PAPEL":
            return "TESOURA"
        if jogador_move == "TESOURA":
            return "PEDRA"
        return "INDEFINIDO"
