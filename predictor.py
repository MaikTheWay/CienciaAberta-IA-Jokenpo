# DOC;
# Mantém um histórico das últimas jogadas do adversário.
# Implementa um modelo de Markov (ou rede neural simples) que prevê a próxima jogada baseado no padrão anterior.
# -> Exemplo: se o histórico for ["pedra", "papel"], a IA prevê qual a jogada mais provável em seguida.
#    Atualiza suas probabilidades sempre que a previsão erra (aprendizado contínuo).

from collections import deque
from typing import List, Tuple, Optional
import time


class Predictor:
    """
    Coleta classificações ao longo do tempo e faz uma decisão estável
    com base nos últimos 0.5 segundos do timer.
    """

    def __init__(self, classifier, window_seconds: float = 0.5):
        self.classifier = classifier
        self.window_seconds = float(window_seconds)
        self.samples = deque()  # cada item: (timestamp, classe, confianca)

    def clear(self):
        self.samples.clear()

    def observe(self, landmarks, handedness: Optional[str] = None, timestamp: Optional[float] = None):
        """
        Avalia um frame e armazena o resultado.
        """
        if timestamp is None:
            timestamp = time.time()

        if not landmarks or len(landmarks) < 21:
            return None

        classe, conf, _ = self.classifier.classify(landmarks, handedness)
        if classe in (0, 1, 2):
            self.samples.append((timestamp, classe, conf))

        self._prune(timestamp)
        return classe, conf

    def _prune(self, now: Optional[float] = None):
        if now is None:
            now = time.time()

        while self.samples and (now - self.samples[0][0]) > self.window_seconds:
            self.samples.popleft()

    def should_focus(self, remaining_time: float) -> bool:
        """
        Quando o timer entra na janela final, o sistema deve focar na estabilização da leitura.
        """
        return remaining_time <= self.window_seconds

    def predict_final(self) -> Tuple[int, float]:
        """
        Faz uma votação ponderada nos frames coletados na janela final.
        Retorna (classe, score).
        """
        if not self.samples:
            return -1, 0.0

        scores = {0: 0.0, 1: 0.0, 2: 0.0}
        now = time.time()

        # peso maior para frames mais recentes e com maior confiança
        for ts, classe, conf in self.samples:
            age = now - ts
            recency_weight = max(0.25, 1.0 - (age / self.window_seconds))
            scores[classe] += conf * recency_weight

        classe_final = max(scores, key=scores.get)
        score_final = scores[classe_final]

        return classe_final, score_final

    def predict_current(self, landmarks, handedness: Optional[str] = None):
        """
        Previsão instantânea, útil para overlay na tela.
        """
        if not landmarks or len(landmarks) < 21:
            return -1, 0.0

        classe, conf, _ = self.classifier.classify(landmarks, handedness)
        return classe, conf