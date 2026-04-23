# =============================================================================
# game_logic.py
# =============================================================================
# Controla o estado do jogo, timer e decisões de vitória/derrota.
# =============================================================================

from enum import Enum, auto
import time
from typing import Optional, Dict, Any

from timer import CountdownTimer
from predictor import Predictor
from gesture_classifier import BrainJokenpo


class GameState(Enum):
    """Estados possíveis do jogo."""
    WAIT_HAND = auto()
    COUNTDOWN = auto()
    RESULT = auto()


def result_game(jogador: str, ia: str) -> str:
    """Determina o resultado do jogo."""
    if jogador == "INDEFINIDO" or ia == "INDEFINIDO":
        return "SEM RESULTADO"

    if jogador == ia:
        return "EMPATE"

    vence = {
        "PEDRA": "TESOURA",
        "PAPEL": "PEDRA",
        "TESOURA": "PAPEL",
    }

    if vence[jogador] == ia:
        return "JOGADOR VENCEU"
    return "IA VENCEU"


class GameLogic:
    """
    Controla o estado do jogo e coordena predição.
    """

    def __init__(
        self,
        timer_seconds: float = 3.0,
        final_window_seconds: float = 0.5
    ):
        self.timer = CountdownTimer(
            duration_seconds=timer_seconds,
            final_window_seconds=final_window_seconds,
        )

        self.classifier = BrainJokenpo()
        self.predictor = Predictor(
            classifier=self.classifier,
            window_seconds=final_window_seconds
        )

        self.state = GameState.WAIT_HAND
        self.player_move = "INDEFINIDO"
        self.ai_move = "INDEFINIDO"
        self.result = "SEM RESULTADO"

        self.hand_visible_counter = 0
        self.round_start_time: Optional[float] = None
        self.current_prediction: Optional[Dict[str, Any]] = None

        # Estatísticas
        self.total_rounds = 0
        self.player_wins = 0
        self.ai_wins = 0
        self.draws = 0

    def reset_round(self):
        """Reseta o estado para uma nova rodada."""
        self.timer.reset()
        self.predictor.clear()
        self.state = GameState.WAIT_HAND
        self.player_move = "INDEFINIDO"
        self.ai_move = "INDEFINIDO"
        self.result = "SEM RESULTADO"
        self.hand_visible_counter = 0
        self.round_start_time = None
        self.current_prediction = None

    def start_round(self):
        """Inicia uma nova rodada."""
        self.timer.start()
        self.predictor.clear()
        self.state = GameState.COUNTDOWN
        self.player_move = "INDEFINIDO"
        self.ai_move = "INDEFINIDO"
        self.result = "SEM RESULTADO"
        self.hand_visible_counter = 0
        self.round_start_time = time.time()

    def _hand_is_stable(self, hand_visible: bool) -> bool:
        """Verifica se a mão está estável para começar."""
        if hand_visible:
            self.hand_visible_counter += 1
        else:
            self.hand_visible_counter = 0
        return self.hand_visible_counter >= 6

    def update(self, hand_visible: bool, landmarks, handedness) -> Dict[str, Any]:
        """Atualiza o estado do jogo a cada frame."""
        now = time.time()

        if self.state == GameState.WAIT_HAND:
            if self._hand_is_stable(hand_visible):
                self.start_round()

        elif self.state == GameState.COUNTDOWN:
            if hand_visible and landmarks:
                self.predictor.observe(landmarks, handedness=handedness, timestamp=now)
                self.current_prediction = self.predictor.predict_realtime()

            if self.timer.finished():
                self._finalize_round()

        return self.get_snapshot()

    def _finalize_round(self):
        """Finaliza a rodada e determina o resultado."""
        classe_final, _ = self.predictor.predict_final()
        self.player_move = self.classifier.label_to_text(classe_final)

        counter, _, _ = self.predictor.get_counter_move(classe_final)
        self.ai_move = self.classifier.label_to_text(counter)

        self.result = result_game(self.player_move, self.ai_move)

        self.total_rounds += 1
        if "JOGADOR VENCEU" in self.result:
            self.player_wins += 1
        elif "IA VENCEU" in self.result:
            self.ai_wins += 1
        else:
            self.draws += 1

        self.state = GameState.RESULT

    def get_snapshot(self) -> Dict[str, Any]:
        """Retorna snapshot completo do estado atual."""
        return {
            "state": self.state,
            "timer_text": self.timer.visible_text(),
            "timer_remaining": self.timer.remaining(),
            "in_final_window": self.timer.in_final_window(),
            "player_move": self.player_move,
            "ai_move": self.ai_move,
            "result": self.result,
            "current_prediction": self.current_prediction,
            "stats": self.get_statistics(),
        }

    def get_statistics(self) -> Dict[str, Any]:
        """Retorna estatísticas do jogo."""
        total = self.total_rounds if self.total_rounds > 0 else 1
        return {
            "total_rounds": self.total_rounds,
            "player_wins": self.player_wins,
            "ai_wins": self.ai_wins,
            "draws": self.draws,
            "player_win_rate": self.player_wins / total * 100,
            "ai_win_rate": self.ai_wins / total * 100,
        }
