# DOC;
# Decide quem ganha com base nas duas jogadas.
# Controla pontuação da IA e do usuário.

from enum import Enum, auto
import time

from timer import CountdownTimer
from predictor import Predictor
from gesture_classifier import BrainJokenpo


class GameState(Enum):
    WAIT_HAND = auto()
    COUNTDOWN = auto()
    RESULT = auto()


def result_game(jogador: str, ia: str) -> str:
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
    Controla o estado do jogo e o fechamento da jogada no fim do timer.
    """

    def __init__(self, timer_seconds: float = 3.0, final_window_seconds: float = 0.5):
        self.timer = CountdownTimer(
            duration_seconds=timer_seconds,
            final_window_seconds=final_window_seconds,
        )
        self.classifier = BrainJokenpo()
        self.predictor = Predictor(self.classifier, window_seconds=final_window_seconds)

        self.state = GameState.WAIT_HAND
        self.player_move = "INDEFINIDO"
        self.ai_move = "INDEFINIDO"
        self.result = "SEM RESULTADO"

        self.hand_visible_counter = 0

    def reset_round(self):
        self.timer.reset()
        self.predictor.clear()
        self.state = GameState.WAIT_HAND
        self.player_move = "INDEFINIDO"
        self.ai_move = "INDEFINIDO"
        self.result = "SEM RESULTADO"
        self.hand_visible_counter = 0

    def start_round(self):
        self.timer.start()
        self.predictor.clear()
        self.state = GameState.COUNTDOWN
        self.player_move = "INDEFINIDO"
        self.ai_move = "INDEFINIDO"
        self.result = "SEM RESULTADO"

    def _hand_is_stable(self, hand_visible: bool) -> bool:
        """
        Exige alguns frames consecutivos com mão visível antes de iniciar o timer.
        Isso evita disparo acidental.
        """
        if hand_visible:
            self.hand_visible_counter += 1
        else:
            self.hand_visible_counter = 0

        return self.hand_visible_counter >= 6

    def update(self, hand_visible: bool, landmarks, handedness):
        """
        Atualiza o estado do jogo.
        Deve ser chamado a cada frame.
        """
        now = time.time()

        if self.state == GameState.WAIT_HAND:
            if self._hand_is_stable(hand_visible):
                self.start_round()

        elif self.state == GameState.COUNTDOWN:
            if hand_visible and landmarks:
                self.predictor.observe(landmarks, handedness=handedness, timestamp=now)

            # No último meio segundo, já vamos focando totalmente na janela final
            if self.timer.in_final_window():
                pass

            if self.timer.finished():
                classe_final, _ = self.predictor.predict_final()
                self.player_move = self.classifier.label_to_text(classe_final)
                self.ai_move = self.classifier.counter_move(self.player_move)
                self.result = result_game(self.player_move, self.ai_move)
                self.state = GameState.RESULT

        elif self.state == GameState.RESULT:
            pass

        return self.get_snapshot()

    def get_snapshot(self):
        return {
            "state": self.state,
            "timer_text": self.timer.visible_text(),
            "timer_remaining": self.timer.remaining(),
            "in_final_window": self.timer.in_final_window(),
            "player_move": self.player_move,
            "ai_move": self.ai_move,
            "result": self.result,
        }