# =============================================================================
# main.py
# =============================================================================
# Ponto de entrada do jogo. Orquestra a GUI medieval e a lógica do jogo.
# =============================================================================

from __future__ import annotations
import sys
from pathlib import Path

# Garantir que o diretório base está no path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from game_logic import GameLogic
from hand_detector import HandDetector
from gui import MedievalJokenpoGUI


def main():
    """Inicializa e executa o jogo."""
    detector = HandDetector()
    game = GameLogic(timer_seconds=3.0, final_window_seconds=0.2)
    
    app = MedievalJokenpoGUI(detector=detector, game=game)
    app.run()


if __name__ == "__main__":
    main()
