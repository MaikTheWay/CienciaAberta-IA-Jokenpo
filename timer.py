# DOC;
# Função que executa uma contagem regressiva na tela: 3, 2, 1!
# Sincroniza a captura da jogada do usuário com o momento exato do 1!.

import time


class CountdownTimer:
    """
    Timer simples com janela final configurável.
    """

    def __init__(self, duration_seconds: float = 3.0, final_window_seconds: float = 0.5):
        self.duration_seconds = float(duration_seconds)
        self.final_window_seconds = float(final_window_seconds)
        self._start_time = None

    def start(self):
        self._start_time = time.time()

    def reset(self):
        self._start_time = None

    def is_running(self) -> bool:
        return self._start_time is not None

    def elapsed(self) -> float:
        if self._start_time is None:
            return 0.0
        return time.time() - self._start_time

    def remaining(self) -> float:
        return max(0.0, self.duration_seconds - self.elapsed())

    def finished(self) -> bool:
        return self.is_running() and self.remaining() <= 0.0

    def in_final_window(self) -> bool:
        if not self.is_running():
            return False
        rem = self.remaining()
        return 0.0 < rem <= self.final_window_seconds

    def progress(self) -> float:
        if not self.is_running():
            return 0.0
        return min(1.0, self.elapsed() / self.duration_seconds)

    def visible_text(self) -> str:
        if not self.is_running():
            return "PRONTO"

        rem = self.remaining()
        if rem > 2:
            return "3"
        if rem > 1:
            return "2"
        if rem > 0:
            return "1"
        return "JÁ!"