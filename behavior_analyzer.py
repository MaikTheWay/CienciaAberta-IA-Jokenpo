# =============================================================================
# behavior_analyzer.py
# =============================================================================
# Mecanismo adaptativo para análise de padrões comportamentais do jogador.
# Identifica sequências e tendências nas jogadas para melhorar predições.
#
# Funcionalidades:
# - Registro de histórico de jogadas
# - Análise de transições entre gestos (matriz de Markov)
# - Detecção de padrões sequenciais
# - Identificação de viés comportamental
# - Ajuste adaptativo de probabilidades
#
# Alberto Seleto de Souza / Marcos Alcino Ribeiro Cussioli
# =============================================================================

import numpy as np
from collections import deque, Counter
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass, field
from enum import Enum
import json
import os
import time


class GestureClass(Enum):
    """Enum para classes de gestos."""
    ROCK = 0
    PAPER = 1
    SCISSORS = 2
    UNKNOWN = -1

    @staticmethod
    def from_string(s: str) 'GestureClass':
        s = s.upper()
        if 'PEDRA' in s or 'ROCK' in s:
            return GestureClass.ROCK
        if 'PAPEL' in s or 'PAPER' in s:
            return GestureClass.PAPER
        if 'TESOURA' in s or 'SCISSORS' in s:
            return GestureClass.SCISSORS
        return GestureClass.UNKNOWN

    @staticmethod
    def from_int(i: int) 'GestureClass':
        if i == 0:
            return GestureClass.ROCK
        if i == 1:
            return GestureClass.PAPER
        if i == 2:
            return GestureClass.SCISSORS
        return GestureClass.UNKNOWN


@dataclass
class GameRecord:
    """Registro de uma jogada completa."""
    player_gesture: GestureClass
    ai_gesture: GestureClass
    result: str
    timestamp: float
    player_win: bool
    round_duration: float = 0.0


@dataclass
class TransitionStats:
    """Estatísticas de transição entre gestos."""
    from_gesture: GestureClass
    to_gesture: GestureClass
    count: int = 0
    probability: float = 0.0


@dataclass
class PatternMatch:
    """Resultado de um padrão encontrado."""
    pattern: List[int]  # Sequência de gestos
    occurrences: int
    frequency: float
    next_probabilities: Dict[int, float]


class BehaviorAnalyzer:
    """
    Analisador de comportamento do jogador.

    Mantém histórico de jogadas e calcula estatísticas para identificar
    padrões e tendências que podem ser usados para melhorar as predições.

    Principais funcionalidades:
    1. Matriz de transição de Markov (próxima jogada baseada na anterior)
    2. Análise de sequências (n-grams)
    3. Detecção de viés (qual gesto o jogador prefere)
    4. Predição baseada em histórico
    """

    # Constantes
    CLASS_NAMES = ['PEDRA', 'PAPEL', 'TESOURA']
    NUM_CLASSES = 3

    def __init__(
        self,
        max_history: int = 100,
        markov_order: int = 1,
        pattern_min_length: int = 3,
        pattern_max_length: int = 5,
        adaptation_rate: float = 0.15,
        confidence_threshold: float = 0.6
    ):
        """
        Inicializa o analisador comportamental.

        Args:
            max_history: Máximo de registros de jogadas a manter
            markov_order: Ordem do modelo de Markov (1=última jogada, 2=duas últimas)
            pattern_min_length: Comprimento mínimo de padrão a detectar
            pattern_max_length: Comprimento máximo de padrão a detectar
            adaptation_rate: Taxa de adaptação para probabilities (0-1)
            confidence_threshold: Limiar de confiança para usar predição histórica
        """
        self.max_history = max_history
        self.markov_order = markov_order
        self.pattern_min_length = pattern_min_length
        self.pattern_max_length = pattern_max_length
        self.adaptation_rate = adaptation_rate
        self.confidence_threshold = confidence_threshold

        # Histórico de jogadas
        self.game_history: deque[GameRecord] = deque(maxlen=max_history)
        self.player_sequence: deque[int] = deque(maxlen=max_history)

        # Matriz de transição de Markov
        # transition_matrix[i][j] = P(próxima=j | atual=i)
        self.transition_matrix: np.ndarray = np.ones(
            (self.NUM_CLASSES, self.NUM_CLASSES)
        ) / self.NUM_CLASSES  # Inicialização uniforme

        # Contadores para matriz
        self.transition_counts: np.ndarray = np.zeros(
            (self.NUM_CLASSES, self.NUM_CLASSES)
        )

        # Estatísticas de n-grams
        self.ngram_counts: Dict[int, Counter] = {
            n: Counter() for n in range(pattern_min_length, pattern_max_length + 1)
        }

        # Viés do jogador (preferência por certos gestos)
        self.player_bias: np.ndarray = np.ones(self.NUM_CLASSES) / self.NUM_CLASSES

        # Contadores de gestos
        self.gesture_counts: np.ndarray = np.zeros(self.NUM_CLASSES)

        # Estatísticas de win/loss
        self.win_streak: int = 0
        self.loss_streak: int = 0
        self.current_streak_type: Optional[str] = None  # 'win' ou 'loss'

        # Timestamps
        self.creation_time = time.time()
        self.last_update_time = time.time()

    def record_game(self, player_gesture: str, ai_gesture: str, result: str, round_duration: float = 0.0):
        """
        Registra uma jogada completa no histórico.

        Args:
            player_gesture: Gesto do jogador ('PEDRA', 'PAPEL', 'TESOURA')
            ai_gesture: Gesto da IA
            result: Resultado da rodada
            round_duration: Duração da rodada em segundos
        """
        player_class = GestureClass.from_string(player_gesture)
        ai_class = GestureClass.from_string(ai_gesture)

        if player_class == GestureClass.UNKNOWN:
            return

        # Determina se jogador venceu
        player_win = 'JOGADOR' in result.upper()

        # Cria registro
        record = GameRecord(
            player_gesture=player_class,
            ai_gesture=ai_class,
            result=result,
            timestamp=time.time(),
            player_win=player_win,
            round_duration=round_duration
        )

        self.game_history.append(record)
        self.player_sequence.append(player_class.value)

        # Atualiza estatísticas
        self._update_transition_matrix(player_class.value)
        self._update_ngrams()
        self._update_player_bias(player_class.value)

        # Atualiza streaks
        if player_win:
            if self.current_streak_type == 'win':
                self.win_streak += 1
            else:
                self.win_streak = 1
                self.loss_streak = 0
                self.current_streak_type = 'win'
        else:
            if self.current_streak_type == 'loss':
                self.loss_streak += 1
            else:
                self.loss_streak = 1
                self.win_streak = 0
                self.current_streak_type = 'loss'

        self.last_update_time = time.time()

    def _update_transition_matrix(self, current_gesture: int):
        """Atualiza matriz de transição com novo dado."""
        if len(self.player_sequence) < 2:
            return

        prev_gesture = self.player_sequence[-2]

        # Incrementa contador
        self.transition_counts[prev_gesture][current_gesture] += 1

        # Recalcula probabilidades com smoothing de Laplace
        row = self.transition_counts[prev_gesture]
        total = row.sum()
        if total > 0:
            self.transition_matrix[prev_gesture] = (row + 1) / (total + self.NUM_CLASSES)
        else:
            self.transition_matrix[prev_gesture] = np.ones(self.NUM_CLASSES) / self.NUM_CLASSES

    def _update_ngrams(self):
        """Atualiza contadores de n-grams."""
        seq = list(self.player_sequence)
        seq_len = len(seq)

        for n in range(self.pattern_min_length, min(self.pattern_max_length + 1, seq_len + 1)):
            # Para cada n-gram possível
            for i in range(seq_len - n + 1):
                ngram = tuple(seq[i:i+n])
                self.ngram_counts[n][ngram] += 1

    def _update_player_bias(self, gesture: int):
        """Atualiza viés do jogador (preferência por gestos)."""
        self.gesture_counts[gesture] += 1

        # Calcula viés com smoothing
        total = self.gesture_counts.sum()
        if total > 0:
            self.player_bias = (self.gesture_counts + 1) / (total + self.NUM_CLASSES)

    def predict_next_gesture(self, use_patterns: bool = True, use_markov: bool = True) -> Tuple[int, float]:
        """
        Prediz o próximo gesto baseado no histórico.

        Combina múltiplas fontes de informação:
        1. Matriz de Markov (baseado na última jogada)
        2. N-grams (padrões sequenciais)
        3. Viés do jogador

        Args:
            use_patterns: Se deve usar detecção de padrões
            use_markov: Se deve usar modelo de Markov

        Returns:
            Tuple (gesture_id, confidence)
        """
        if len(self.player_sequence) < 2:
            # Sem dados suficientes, usa viés uniforme
            return np.random.randint(0, self.NUM_CLASSES), 0.33

        # Combina diferentes fontes de probabilidade
        combined_probs = np.zeros(self.NUM_CLASSES)
        weights = []

        # 1. Probabilidades de Markov
        if use_markov and len(self.player_sequence) >= self.markov_order + 1:
            last_gesture = self.player_sequence[-1]
            markov_probs = self.transition_matrix[last_gesture]
            combined_probs += markov_probs
            weights.append(0.5)

        # 2. N-grams (padrões mais longos têm mais peso)
        if use_patterns and len(self.player_sequence) >= self.pattern_min_length:
            pattern_probs = self._get_pattern_probabilities()
            if pattern_probs is not None:
                combined_probs += pattern_probs
                weights.append(0.35)

        # 3. Viés do jogador
        combined_probs += self.player_bias
        weights.append(0.15)

        # Normaliza pela soma dos pesos
        if weights:
            total_weight = sum(weights)
            combined_probs = combined_probs / total_weight

        # Verifica confiança
        max_prob = combined_probs.max()
        if max_prob < self.confidence_threshold:
            # Baixa confiança, usa distribuição uniforme ponderada
            confidence = max_prob
        else:
            confidence = max_prob

        predicted_gesture = int(np.argmax(combined_probs))

        return predicted_gesture, float(confidence)

    def _get_pattern_probabilities(self) -> Optional[np.ndarray]:
        """
        Obtém probabilidades baseadas em padrões n-gram detectados.

        Procura o maior n-gram que corresponde ao final da sequência
        e usa as transições observadas após esse padrão.

        Returns:
            Array de probabilidades ou None se nenhum padrão encontrado
        """
        seq = list(self.player_sequence)
        seq_len = len(seq)

        # Procura do maior n-gram para o menor
        for n in range(min(self.pattern_max_length, seq_len), self.pattern_min_length - 1, -1):
            # Pega os últimos n gestos
            pattern = tuple(seq[-n:])

            # Verifica se existe contagem para este padrão
            if pattern in self.ngram_counts[n]:
                # Procura contextos que começam com este padrão
                prefix_counts = Counter()
                total = 0

                for full_pattern, count in self.ngram_counts[n].items():
                    if full_pattern[:-1] == pattern:
                        next_gesture = full_pattern[-1]
                        prefix_counts[next_gesture] = count
                        total += count

                if total > 0:
                    probs = np.zeros(self.NUM_CLASSES)
                    for gesture, count in prefix_counts.items():
                        probs[gesture] = count / total
                    return probs

        return None

    def get_counter_prediction(self, base_prediction: int) -> Tuple[int, float, str]:
        """
        Retorna a jogada que contra-ataca a predição.

        Args:
            base_prediction: Gesto que acreditamos que o jogador vai fazer

        Returns:
            Tuple (counter_gesture, confidence, method)
        """
        # Estratégia padrão: contra-ataque direto
        counter_map = {0: 1, 1: 2, 2: 0}  # Pedra->Papel, Papel->Tesoura, Tesoura->Pedra

        # Se temos alta confiança no histórico, ajusta
        predicted, historical_confidence = self.predict_next_gesture()

        if historical_confidence > self.confidence_threshold and predicted == base_prediction:
            # Histórico concorda com predição base
            counter = counter_map[predicted]
            confidence = historical_confidence * self.adaptation_rate + 0.5
            method = "historical_adapted"
        else:
            # Usa predição do modelo temporal ou marca com baixa confiança
            counter = counter_map.get(base_prediction, np.random.randint(0, 3))
            confidence = 0.5
            method = "direct_counter"

        return counter, float(confidence), method

    def analyze_player_tendencies(self) -> Dict:
        """
        Retorna análise completa das tendências do jogador.

        Returns:
            Dicionário com estatísticas comportamentais
        """
        if len(self.game_history) < 3:
            return {
                'has_enough_data': False,
                'message': 'Dados insuficientes para análise'
            }

        total_games = len(self.game_history)

        # Contagens de gestos
        gesture_names = ['rock', 'paper', 'scissors']
        gesture_stats = {
            name: {
                'count': int(self.gesture_counts[i]),
                'percentage': float(self.gesture_counts[i] / total_games * 100)
            }
            for i, name in enumerate(gesture_names)
        }

        # Tendência de transição
        last_gesture = self.player_sequence[-1] if self.player_sequence else -1
        next_probs = {}
        if last_gesture >= 0:
            for i, name in enumerate(gesture_names):
                next_probs[name] = float(self.transition_matrix[last_gesture][i])

        # Padrões detectados
        detected_patterns = []
        for n in range(self.pattern_min_length, self.pattern_max_length + 1):
            if len(self.ngram_counts[n]) > 0:
                # Top 3 padrões mais frequentes
                top = self.ngram_counts[n].most_common(3)
                for pattern, count in top:
                    if count >= 2:  # Só inclui padrões com pelo menos 2 ocorrências
                        pattern_str = ' -> '.join([gesture_names[g] for g in pattern])
                        detected_patterns.append({
                            'pattern': pattern_str,
                            'length': n,
                            'occurrences': count,
                            'frequency': count / total_games * 100
                        })

        return {
            'has_enough_data': True,
            'total_games_analyzed': total_games,
            'streak_info': {
                'current_type': self.current_streak_type,
                'wins': self.win_streak,
                'losses': self.loss_streak
            },
            'gesture_preferences': gesture_stats,
            'next_gesture_probabilities': next_probs,
            'detected_patterns': detected_patterns[:5],  # Top 5
            'average_round_duration': self._get_average_duration()
        }

    def _get_average_duration(self) -> float:
        """Calcula duração média das rodadas."""
        if not self.game_history:
            return 0.0
        durations = [r.round_duration for r in self.game_history if r.round_duration > 0]
        return np.mean(durations) if durations else 0.0

    def get_confidence_level(self) -> Tuple[str, float]:
        """
        Retorna nível de confiança na análise.

        Returns:
            Tuple (level, score) onde level é 'low', 'medium' ou 'high'
        """
        samples = len(self.game_history)

        if samples < 5:
            return 'low', samples / 10
        elif samples < 20:
            return 'medium', samples / 20
        else:
            # Calcula entropia do viés (mais uniforme = mais dados necessários)
            entropy = -np.sum(self.player_bias * np.log(self.player_bias + 1e-10))
            max_entropy = np.log(self.NUM_CLASSES)
            confidence = 1 - (entropy / max_entropy)

            return 'high', float(confidence * 0.8 + 0.2)

    def reset(self):
        """Reseta todo o histórico e estatísticas."""
        self.game_history.clear()
        self.player_sequence.clear()
        self.transition_matrix = np.ones((self.NUM_CLASSES, self.NUM_CLASSES)) / self.NUM_CLASSES
        self.transition_counts = np.zeros((self.NUM_CLASSES, self.NUM_CLASSES))
        self.ngram_counts = {n: Counter() for n in range(self.pattern_min_length, self.pattern_max_length + 1)}
        self.player_bias = np.ones(self.NUM_CLASSES) / self.NUM_CLASSES
        self.gesture_counts = np.zeros(self.NUM_CLASSES)
        self.win_streak = 0
        self.loss_streak = 0
        self.current_streak_type = None

    def save_state(self, filepath: str):
        """Salva o estado do analisador em arquivo."""
        state = {
            'max_history': self.max_history,
            'markov_order': self.markov_order,
            'transition_matrix': self.transition_matrix.tolist(),
            'transition_counts': self.transition_counts.tolist(),
            'gesture_counts': self.gesture_counts.tolist(),
            'player_bias': self.player_bias.tolist(),
            'win_streak': self.win_streak,
            'loss_streak': self.loss_streak,
            'current_streak_type': self.current_streak_type,
            'game_history': [
                {
                    'player_gesture': r.player_gesture.value,
                    'ai_gesture': r.ai_gesture.value,
                    'result': r.result,
                    'timestamp': r.timestamp,
                    'player_win': r.player_win
                }
                for r in self.game_history
            ]
        }

        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)

        print(f"[BehaviorAnalyzer] Estado salvo em: {filepath}")

    def load_state(self, filepath: str) -> bool:
        """Carrega o estado do analisador de arquivo."""
        if not os.path.exists(filepath):
            print(f"[BehaviorAnalyzer] Arquivo não encontrado: {filepath}")
            return False

        try:
            with open(filepath, 'r') as f:
                state = json.load(f)

            self.max_history = state.get('max_history', self.max_history)
            self.markov_order = state.get('markov_order', self.markov_order)
            self.transition_matrix = np.array(state['transition_matrix'])
            self.transition_counts = np.array(state['transition_counts'])
            self.gesture_counts = np.array(state['gesture_counts'])
            self.player_bias = np.array(state['player_bias'])
            self.win_streak = state.get('win_streak', 0)
            self.loss_streak = state.get('loss_streak', 0)
            self.current_streak_type = state.get('current_streak_type')

            print(f"[BehaviorAnalyzer] Estado carregado de: {filepath}")
            return True

        except Exception as e:
            print(f"[BehaviorAnalyzer] Erro ao carregar estado: {e}")
            return False

    def get_summary(self) -> str:
        """Retorna string com resumo do analisador."""
        samples = len(self.game_history)
        confidence_level, confidence_score = self.get_confidence_level()

        return f"""
=== Behavior Analyzer Summary ===
Amostras coletadas: {samples}
Nível de confiança: {confidence_level} ({confidence_score:.2f})
Sequências Markov: {len(self.player_sequence)}

Viés do jogador:
  PEDRA: {self.player_bias[0]:.1%}
  PAPEL: {self.player_bias[1]:.1%}
  TESOURA: {self.player_bias[2]:.1%}

Streak atual: {self.current_streak_type or 'N/A'} ({self.win_streak if self.current_streak_type == 'win' else self.loss_streak})
"""


class AdaptivePredictor:
    """
    Preditor adaptativo que combina modelo temporal com análise comportamental.

    Este predidor usa:
    1. Modelo LSTM/GRU para predição imediata baseada em sequência de frames
    2. Analyzer comportamental para ajuste baseado em histórico de jogadas
    3. Fusão adaptativa das duas fontes de informação
    """

    def __init__(
        self,
        temporal_predictor,  # Tipo: TemporalPredictor
        behavior_analyzer,     # Tipo: BehaviorAnalyzer
        temporal_weight: float = 0.7,
        behavior_weight: float = 0.3,
        min_confidence_for_override: float = 0.75
    ):
        """
        Inicializa o preditor adaptativo.

        Args:
            temporal_predictor: Instância do TemporalPredictor
            behavior_analyzer: Instância do BehaviorAnalyzer
            temporal_weight: Peso do modelo temporal na fusão
            behavior_weight: Peso da análise comportamental na fusão
            min_confidence_for_override: Confiança mínima para sobrepor predição
        """
        self.temporal = temporal_predictor
        self.behavior = behavior_analyzer
        self.temporal_weight = temporal_weight
        self.behavior_weight = behavior_weight
        self.min_confidence_for_override = min_confidence_for_override

    def predict(self, sequence: np.ndarray) -> Tuple[int, float, str]:
        """
        Faz predição adaptativa combinando modelo temporal e análise comportamental.

        Args:
            sequence: Array numpy com sequência de landmarks

        Returns:
            Tuple (gesture_id, confidence, source)
            - source: 'temporal', 'behavior', 'fused' ou 'fallback'
        """
        # 1. Predição temporal
        temporal_result = self.temporal.predict(sequence)

        if temporal_result.class_id < 0:
            # Sem dados suficientes, usa comportamento ou fallback
            return self._fallback_prediction()

        # 2. Predição comportamental
        behavior_pred, behavior_conf = self.behavior.predict_next_gesture()

        # 3. Decide estratégia de fusão
        temporal_confidence = temporal_result.confidence
        behavior_confidence = behavior_conf

        if temporal_confidence >= self.min_confidence_for_override:
            # Alta confiança no temporal, usa principalmente ele
            return temporal_result.class_id, temporal_confidence, 'temporal'

        elif behavior_confidence >= self.min_confidence_for_override and temporal_confidence < 0.5:
            # Baixa confiança temporal mas alta comportamental
            # Considera sobrepor
            if behavior_pred != temporal_result.class_id:
                counter, _, _ = self.behavior.get_counter_prediction(temporal_result.class_id)
                return counter, behavior_confidence * 0.8, 'behavior_adjusted'

            return behavior_pred, behavior_confidence, 'behavior'

        else:
            # Fusão suave
            fused_probs = (
                temporal_result.probabilities * self.temporal_weight +
                self._get_behavior_probs() * self.behavior_weight
            )

            # Normaliza
            fused_probs = fused_probs / fused_probs.sum()

            final_class = int(np.argmax(fused_probs))
            final_conf = float(fused_probs[final_class])

            return final_class, final_conf, 'fused'

    def _get_behavior_probs(self) -> np.ndarray:
        """Obtém probabilidades do modelo comportamental."""
        pred, _ = self.behavior.predict_next_gesture()
        probs = np.zeros(3)
        probs[pred] = 0.6  # Maior probabilidade para predição
        # Distribui restante uniformemente
        other_indices = [i for i in range(3) if i != pred]
        probs[other_indices[0]] = 0.2
        probs[other_indices[1]] = 0.2
        return probs

    def _fallback_prediction(self) -> Tuple[int, float, str]:
        """Predição de fallback quando não há dados."""
        return np.random.randint(0, 3), 0.33, 'fallback'

    def reset(self):
        """Reseta ambos os componentes."""
        self.temporal.reset_history()
        # Não reseta behavior para manter histórico
