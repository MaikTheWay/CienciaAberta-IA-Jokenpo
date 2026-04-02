# =============================================================================
# sequence_buffer.py
# =============================================================================
# Buffer FIFO temporal para armazenamento de sequência de landmarks da mão.
# Mantém os últimos ~0.3-0.5 segundos de frames para análise sequencial.
#
# Funcionalidades:
# - Armazenamento de sequência temporal de 21 landmarks (x, y, z)
# - Buffer circular FIFO com tamanho configurável
# - Cálculo de features derivadas (velocidade, aceleração)
# - Aplicação de pesos exponenciais para frames mais recentes
# - Normalização de dados para entrada do modelo LSTM/GRU
#
# Alberto Seleto de Souza / Marcos Alcino Ribeiro Cussioli
# =============================================================================

import numpy as np
from collections import deque
from typing import List, Tuple, Optional, Dict, Deque
import time


class SequenceBuffer:
    """
    Buffer FIFO circular que mantém uma sequência temporal de landmarks da mão.

    O buffer armazena os últimos N frames de landmarks (21 pontos x,y,z = 63 features)
    permitindo análise de padrões temporais como abertura/fechamento de mão,
    velocidade de movimento e transições entre gestos.

    Parâmetros:
        buffer_duration: Duração em segundos a manter no buffer (default: 0.4s)
        fps: FPS estimado da captura (default: 30)
        include_velocity: Se True, calcula features de velocidade (default: True)
        include_acceleration: Se True, calcula features de aceleração (default: False)
        weight_scheme: Esquema de pesagem ('exponential', 'linear', 'uniform')
    """

    LANDMARK_COUNT = 21
    COORDS_PER_LANDMARK = 3  # x, y, z

    def __init__(
        self,
        buffer_duration: float = 0.4,
        fps: float = 30.0,
        include_velocity: bool = True,
        include_acceleration: bool = False,
        weight_scheme: str = 'exponential',
        decay_rate: float = 2.5
    ):
        self.buffer_duration = buffer_duration
        self.fps = fps
        self.include_velocity = include_velocity
        self.include_acceleration = include_acceleration
        self.weight_scheme = weight_scheme
        self.decay_rate = decay_rate

        # Calcula tamanho do buffer baseado na duração
        self.max_frames = max(5, int(buffer_duration * fps))

        # Buffers circulares para dados
        self._landmarks_buffer: Deque[List[float]] = deque(maxlen=self.max_frames)
        self._timestamps: Deque[float] = deque(maxlen=self.max_frames)

        # Frame anterior para cálculo de velocidade
        self._prev_landmarks: Optional[np.ndarray] = None

        # Normalização - inicializada lazily
        self._normalization_stats: Optional[Dict[str, np.ndarray]] = None
        self._is_initialized = False

    def clear(self):
        """Limpa todos os buffers."""
        self._landmarks_buffer.clear()
        self._timestamps.clear()
        self._prev_landmarks = None

    def add_frame(self, landmarks: List[Tuple[float, float, float]], timestamp: Optional[float] = None) -> bool:
        """
        Adiciona um novo frame de landmarks ao buffer.

        Args:
            landmarks: Lista de 21 tuples (x, y, z) dos landmarks da mão
            timestamp: Timestamp do frame (default: time.time())

        Returns:
            True se o frame foi adicionado com sucesso, False se dados inválidos
        """
        if timestamp is None:
            timestamp = time.time()

        # Validação dos landmarks
        if not landmarks or len(landmarks) < self.LANDMARK_COUNT:
            return False

        # Converte para array flatten [x1,y1,z1,x2,y2,z2,...]
        flat_landmarks = []
        for lm in landmarks[:self.LANDMARK_COUNT]:
            flat_landmarks.extend([lm[0], lm[1], lm[2]])

        self._landmarks_buffer.append(flat_landmarks)
        self._timestamps.append(timestamp)

        self._is_initialized = len(self._landmarks_buffer) >= 3
        return True

    def get_sequence_length(self) -> int:
        """Retorna o número de frames atualmente no buffer."""
        return len(self._landmarks_buffer)

    def get_raw_sequence(self) -> np.ndarray:
        """
        Retorna a sequência raw de landmarks como array numpy.

        Returns:
            Array shape (seq_len, 63) com os landmarks flatten
        """
        if not self._landmarks_buffer:
            return np.array([])

        return np.array(list(self._landmarks_buffer))

    def get_sequence_with_features(self) -> np.ndarray:
        """
        Retorna sequência de landmarks com features derivadas calculadas.

        Features por frame:
        - 63 valores de landmarks (x, y, z * 21 pontos)
        - 63 valores de velocidade (se include_velocity=True)
        - 63 valores de aceleração (se include_acceleration=True)

        Returns:
            Array numpy com shape (seq_len, num_features)
        """
        raw = self.get_raw_sequence()
        if raw.size == 0:
            return np.array([])

        seq_len = raw.shape[0]

        # Começa com landmarks raw
        features_list = [raw]

        # Adiciona velocidade se solicitado
        if self.include_velocity and seq_len > 1:
            velocity = self._compute_velocity(raw)
            features_list.append(velocity)

        # Adiciona aceleração se solicitado
        if self.include_acceleration and seq_len > 2:
            acceleration = self._compute_acceleration(raw)
            features_list.append(acceleration)

        # Concatena todas as features
        return np.concatenate(features_list, axis=1)

    def _compute_velocity(self, landmarks: np.ndarray) -> np.ndarray:
        """
        Calcula a velocidade de mudança dos landmarks entre frames.

        Args:
            landmarks: Array shape (seq_len, 63)

        Returns:
            Array shape (seq_len, 63) com velocidades
        """
        seq_len = landmarks.shape[0]
        velocity = np.zeros_like(landmarks)

        # Calcula diferenças para frames consecutivos
        for i in range(1, seq_len):
            dt = 1.0 / self.fps
            velocity[i] = (landmarks[i] - landmarks[i-1]) / dt

        # Primeiro frame usa mesma velocidade do segundo
        if seq_len > 1:
            velocity[0] = velocity[1]

        return velocity

    def _compute_acceleration(self, landmarks: np.ndarray) -> np.ndarray:
        """
        Calcula a aceleração de mudança dos landmarks.

        Args:
            landmarks: Array shape (seq_len, 63)

        Returns:
            Array shape (seq_len, 63) com acelerações
        """
        seq_len = landmarks.shape[0]
        acceleration = np.zeros_like(landmarks)

        if seq_len > 2:
            dt = 1.0 / self.fps
            for i in range(1, seq_len - 1):
                acceleration[i] = (landmarks[i+1] - 2*landmarks[i] + landmarks[i-1]) / (dt * dt)

            # Frames de borda usam valores do vizinho
            acceleration[0] = acceleration[1]
            acceleration[-1] = acceleration[-2]

        return acceleration

    def get_weighted_sequence(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Retorna sequência com pesos exponenciais para frames mais recentes.

        O esquema de pesagem é crucial para dar maior importância aos frames
        finais da sequência, capturando o momento exato do gesto.

        Peso(t) = exp(-decay_rate * (seq_len - t - 1) / seq_len)

        Returns:
            Tuple de (sequence, weights)
            - sequence: Array numpy (seq_len, features)
            - weights: Array numpy (seq_len,) com pesos normalizados
        """
        sequence = self.get_sequence_with_features()
        if sequence.size == 0:
            return np.array([]), np.array([])

        seq_len = sequence.shape[0]

        # Calcula pesos baseado no esquema
        if self.weight_scheme == 'exponential':
            # Pesos exponenciais decrescentes do passado para presente
            t = np.arange(seq_len)
            weights = np.exp(-self.decay_rate * (seq_len - t - 1) / seq_len)
        elif self.weight_scheme == 'linear':
            # Pesos lineares crescentes
            weights = np.linspace(0.3, 1.0, seq_len)
        else:  # uniform
            weights = np.ones(seq_len)

        # Normaliza pesos para somarem 1
        weights = weights / weights.sum()

        return sequence, weights

    def get_padded_sequence(self, target_length: Optional[int] = None) -> np.ndarray:
        """
        Retorna sequência com padding para tamanho fixo (necessário para LSTM).

        Se a sequência é menor que target_length, preenche com o último frame
        (ou zeros). Se é maior, trunca para os frames mais recentes.

        Args:
            target_length: Tamanho desejado (default: max_frames)

        Returns:
            Array numpy (target_length, features) com padding
        """
        if target_length is None:
            target_length = self.max_frames

        sequence, _ = self.get_weighted_sequence()

        if sequence.size == 0:
            # Retorna array vazio com forma correta
            features_per_frame = self.LANDMARK_COUNT * self.COORDS_PER_LANDMARK
            if self.include_velocity:
                features_per_frame *= 2
            return np.zeros((target_length, features_per_frame))

        seq_len = sequence.shape[0]

        if seq_len < target_length:
            # Padding com último frame
            padding = np.tile(sequence[-1:], (target_length - seq_len, 1))
            padded = np.vstack([sequence, padding])
        else:
            # Trunca para os frames mais recentes
            padded = sequence[-target_length:]

        return padded

    def get_hand_opening_features(self) -> Dict[str, float]:
        """
        Extrai features específicas de abertura/fechamento da mão.

        Useful para detectar transições entre gestos.

        Returns:
            Dicionário com features de abertura
        """
        raw = self.get_raw_sequence()
        if raw.shape[0] < 2:
            return {}

        seq_len = raw.shape[0]

        # Organiza landmarks por dedo para análise
        # Dedos: polegar(0-4), indicador(5-8), médio(9-12), anelar(13-16), mindinho(17-20)
        finger_indices = [
            [0, 1, 2, 3, 4],      # Polegar
            [5, 6, 7, 8],          # Indicador
            [9, 10, 11, 12],       # Médio
            [13, 14, 15, 16],     # Anelar
            [17, 18, 19, 20]      # Mindinho
        ]

        features = {}

        # Calcula distância média do pulso para cada dedo no último frame
        wrist = raw[-1, 0:3]  # Primeiro landmark (pulso)

        for i, finger in enumerate(finger_indices):
            finger_name = ['thumb', 'index', 'middle', 'ring', 'pinky'][i]
            distances = []

            for idx in finger[1:]:  # Exclui CMC/junção com pulso
                lm_idx = idx * 3
                tip = raw[-1, lm_idx:lm_idx+3]
                dist = np.linalg.norm(tip - wrist)
                distances.append(dist)

            features[f'{finger_name}_extension'] = np.mean(distances) if distances else 0.0

        # Calcula velocidade de mudança (abertura/fechamento)
        if seq_len >= 3:
            last_frame = raw[-1]
            first_frame = raw[0]
            frame_diff = last_frame - first_frame
            features['total_change'] = np.linalg.norm(frame_diff)
            features['max_change'] = np.max(np.abs(frame_diff))
        else:
            features['total_change'] = 0.0
            features['max_change'] = 0.0

        # Direção do movimento (abertura ou fechamento)
        if seq_len >= 2:
            frame_diff = raw[-1] - raw[-2]
            features['movement_direction'] = np.linalg.norm(frame_diff)
            features['is_opening'] = 1.0 if features['movement_direction'] > 0.01 else 0.0
        else:
            features['movement_direction'] = 0.0
            features['is_opening'] = 0.0

        return features

    def get_temporal_summary(self) -> Dict[str, any]:
        """
        Retorna um resumo estatístico do buffer atual.

        Returns:
            Dicionário com estatísticas temporais
        """
        raw = self.get_raw_sequence()

        if raw.size == 0:
            return {
                'frame_count': 0,
                'duration_ms': 0.0,
                'fps_actual': 0.0,
                'is_ready': False
            }

        seq_len = raw.shape[0]
        timestamps = list(self._timestamps)

        if seq_len > 1:
            duration = timestamps[-1] - timestamps[0]
            fps_actual = (seq_len - 1) / duration if duration > 0 else 0.0
        else:
            duration = 0.0
            fps_actual = 0.0

        return {
            'frame_count': seq_len,
            'max_frames': self.max_frames,
            'duration_ms': duration * 1000,
            'fps_actual': fps_actual,
            'is_ready': self.is_ready_for_prediction(),
            'buffer_fill_ratio': seq_len / self.max_frames if self.max_frames > 0 else 0
        }

    def is_ready_for_prediction(self) -> bool:
        """
        Verifica se o buffer tem dados suficientes para predição.

        Returns:
            True se tem pelo menos 50% do buffer preenchido
        """
        min_frames = max(3, self.max_frames // 2)
        return len(self._landmarks_buffer) >= min_frames

    def get_latest_prediction_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Retorna dados prontos para predição LSTM.

        Returns:
            Tuple de (sequence_array, weights_array)
        """
        return self.get_weighted_sequence()
