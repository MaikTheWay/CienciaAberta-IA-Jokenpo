# =============================================================================
# predictor.py
# =============================================================================
# Sistema de predição para o jogo Pedra-Papel-Tesoura.
# VERSÃO SIMPLIFICADA: Prioriza classificação baseada em regras.
#
# IMPORTANTE: O modelo temporal LSTM/GRU e fusão foram desabilitados
# porque a classificação por regras (gesture_classifier.py) é a única
# fonte confiável de detecção de gestos no momento.
#
# Alberto Seleto de Souza / Marcos Alcino Ribeiro Cussioli
# =============================================================================

from collections import deque
from typing import List, Tuple, Optional, Dict, Any
import time


class Predictor:
    """
    Sistema de predição simplificado baseado em regras.

    Apenas utiliza o classificador BrainJokenpo para detecção
    de gestos via regras geométricas. Modelos de ML e fusão
    estão desabilitados por não estarem confiáveis.
    """

    def __init__(
        self,
        classifier,
        window_seconds: float = 0.5,
        enable_history: bool = False  # Mantido para possível uso futuro
    ):
        """
        Inicializa o sistema de predição.

        Args:
            classifier: Instância do BrainJokenpo
            window_seconds: Janela temporal para coleta de samples
            enable_history: (Reservado) Para uso futuro
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

        Args:
            landmarks: Lista de 21 tuples (x, y, z)
            handedness: 'Left' ou 'Right'
            timestamp: Timestamp do frame

        Returns:
            Tuple (classe, confiança) se landmarks válidos
        """
        if timestamp is None:
            timestamp = time.time()

        if not landmarks or len(landmarks) < 21:
            return None

        # Classificação baseada em regras (única fonte confiável)
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

    def should_focus(self, remaining_time: float) -> bool:
        """Determina se deve focar na estabilização."""
        return remaining_time <= self.window_seconds

    def predict_final(self) -> Tuple[int, float]:
        """
        Faz predição final usando VOTAÇÃO PONDERADA dos samples.

        Usa apenas o classificador baseado em regras (BrainJokenpo).

        Returns:
            Tuple (classe, score_final)
        """
        if not self.samples:
            return -1, 0.0

        # Votação ponderada por confiança e recência
        scores = {0: 0.0, 1: 0.0, 2: 0.0}
        now = time.time()

        for ts, classe, conf in self.samples:
            # Peso de recência: frames mais recentes pesam mais
            age = now - ts
            recency_weight = max(0.3, 1.0 - (age / self.window_seconds))

            # Score = confiança * peso de recência
            scores[classe] += conf * recency_weight

        # Escolhe classe com maior score
        classe_final = max(scores, key=scores.get)
        score_final = scores[classe_final]

        self._prediction_count += 1
        return classe_final, score_final

    def predict_realtime(self) -> Dict[str, Any]:
        """
        Retorna predição em tempo real (baseada em regras).

        Returns:
            Dicionário com informações de predição
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

        # Último sample
        latest = self.samples[-1]
        result['rule_based'] = {
            'class': latest[1],
            'confidence': latest[2],
            'class_name': self.classifier.label_to_text(latest[1])
        }

        # Estatísticas dos samples na janela
        if len(self.samples) > 1:
            votes = {0: 0, 1: 0, 2: 0}
            for _, classe, _ in self.samples:
                votes[classe] += 1
            result['votes'] = votes

        return result

    def record_game_result(
        self,
        player_gesture: str,
        ai_gesture: str,
        result: str,
        round_duration: float = 0.0
    ):
        """
        Registra resultado de uma rodada (para uso futuro).
        Por enquanto não faz nada - preservado para possível histórico.
        """
        pass  # Reservado para uso futuro

    def get_counter_move(self, prediction: int) -> Tuple[int, float, str]:
        """
        Retorna a jogada que contra-ataca a predição.

        Args:
            prediction: Gesto predito do jogador

        Returns:
            Tuple (counter_gesture, confidence, method)
        """
        # Contra-ataque direto padrão
        counter_map = {0: 1, 1: 2, 2: 0}
        counter = counter_map.get(prediction, 0)
        return counter, 0.5, 'direct'

    def predict_current(self, landmarks, handedness: Optional[str] = None) -> Tuple[int, float]:
        """
        Previsão instantânea (frame atual apenas).

        Args:
            landmarks: Landmarks da mão
            handedness: Lado da mão

        Returns:
            Tuple (class_id, confidence)
        """
        if not landmarks or len(landmarks) < 21:
            return -1, 0.0

        classe, conf, _ = self.classifier.classify(landmarks, handedness)
        return classe, conf

    def predict_temporal(self, landmarks: Optional[List] = None) -> Tuple[int, float, bool]:
        """
        Predição temporal - DESABILITADA.

        Retorna fallback para classificação por regras.
        """
        return -1, 0.0, False

    def get_statistics(self) -> Dict[str, Any]:
        """Retorna estatísticas de predição."""
        return {
            'total_predictions': self._prediction_count,
            'samples_in_buffer': len(self.samples),
            'mode': 'rule_based_only'
        }

    def save_state(self, filepath: str = 'predictor_state'):
        """Salva estado (não implementado na versão simplificada)."""
        pass

    def get_behavior_analysis(self) -> Dict:
        """Retorna análise (não disponível na versão simplificada)."""
        return {'available': False, 'message': 'Análise comportamental desabilitada'}


def create_predictor(
    classifier,
    window_seconds: float = 0.5,
    enable_temporal: bool = False,  # IGNORADO
    enable_behavior: bool = False,  # IGNORADO
    model_type: str = 'lstm'  # IGNORADO
) -> Predictor:
    """
    Factory function para criar um Predictor.

    Args:
        classifier: Instância do classificador
        window_seconds: Janela temporal
        enable_temporal: IGNORADO
        enable_behavior: IGNORADO
        model_type: IGNORADO

    Returns:
        Instância do Predictor
    """
    return Predictor(
        classifier=classifier,
        window_seconds=window_seconds,
        enable_history=False
    )
