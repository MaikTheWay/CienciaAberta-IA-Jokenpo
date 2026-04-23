# =============================================================================
# predictor.py
# =============================================================================
# Sistema de predição simplificado para o jogo Pedra-Papel-Tesoura.
# Utiliza apenas classificação baseada em regras (gesture_classifier.py).
# =============================================================================

from collections import deque
from typing import Tuple, Optional, Dict, Any
import time


class Predictor:
    """
    Sistema de predição simplificado baseado em regras geométricas.
    """

    def __init__(
        self,
        classifier,
        window_seconds: float = 0.5
    ):
        """
        Inicializa o sistema de predição.

        Args:
            classifier: Instância do BrainJokenpo
            window_seconds: Janela temporal para coleta de samples
        """
        self.classifier = classifier
        self.window_seconds = float(window_seconds)
        self.samples = deque()  # cada item: (timestamp, classe, confianca)
        self._prediction_count = 0

    def clear(self):
        """Limpa todos os buffers e histórico."""
        self.samples.clear()
        self._prediction_count = 0

    def observe(
        self,
        landmarks,
        handedness: Optional[str] = None,
        timestamp: Optional[float] = None
    ) -> Optional[Tuple[int, float]]:
        """
        Avalia um frame e armazena o resultado.
        """
        if timestamp is None:
            timestamp = time.time()

        if not landmarks or len(landmarks) < 21:
            return None

        # Classificação baseada em regras
        classe, conf, _ = self.classifier.classify(landmarks, handedness)

        if classe in (0, 1, 2):
            self.samples.append((timestamp, classe, conf))

        self._prune(timestamp)
        return classe, conf

    def _prune(self, now: Optional[float] = None):
        """Remove samples antigos da janela temporal."""
        if now is None:
            now = time.time()

        while self.samples and (now - self.samples[0][0]) > self.window_seconds:
            self.samples.popleft()

    def predict_final(self) -> Tuple[int, float]:
        """
        Faz predição final usando VOTAÇÃO PONDERADA dos samples.
        """
        if not self.samples:
            return -1, 0.0

        scores = {0: 0.0, 1: 0.0, 2: 0.0}
        now = time.time()

        for ts, classe, conf in self.samples:
            age = now - ts
            recency_weight = max(0.3, 1.0 - (age / self.window_seconds))
            scores[classe] += conf * recency_weight

        classe_final = max(scores, key=scores.get)
        score_final = scores[classe_final]

        self._prediction_count += 1
        return classe_final, score_final

    def predict_realtime(self) -> Dict[str, Any]:
        """
        Retorna predição em tempo real.
        """
        result = {
            'timestamp': time.time(),
            'has_data': False,
            'rule_based': {'class': -1, 'confidence': 0.0},
            'samples_count': len(self.samples)
        }

        if not self.samples:
            return result

        result['has_data'] = True
        latest = self.samples[-1]
        result['rule_based'] = {
            'class': latest[1],
            'confidence': latest[2],
            'class_name': self.classifier.label_to_text(latest[1])
        }

        return result

    def get_counter_move(self, prediction: int) -> Tuple[int, float, str]:
        """
        Retorna a jogada que contra-ataca a predição.
        """
        counter_map = {0: 1, 1: 2, 2: 0}
        counter = counter_map.get(prediction, 0)
        return counter, 0.5, 'direct'

    def predict_current(self, landmarks, handedness: Optional[str] = None) -> Tuple[int, float]:
        """
        Previsão instantânea (frame atual apenas).
        """
        if not landmarks or len(landmarks) < 21:
            return -1, 0.0

        classe, conf, _ = self.classifier.classify(landmarks, handedness)
        return classe, conf

    def get_statistics(self) -> Dict[str, Any]:
        """Retorna estatísticas de predição."""
        return {
            'total_predictions': self._prediction_count,
            'samples_in_buffer': len(self.samples)
        }
