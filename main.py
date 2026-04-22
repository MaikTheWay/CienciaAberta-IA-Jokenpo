# =============================================================================
# main.py
# =============================================================================
# Ponto de entrada do jogo.
# Mantém a lógica original disponível em modo console e orquestra a GUI medieval.
#
# Alberto Seleto de Souza
# Marcos Alcino Ribeiro Cussioli
# =============================================================================

from __future__ import annotations

import argparse

import cv2

from game_logic import GameLogic, GameState
from hand_detector import HandDetector


def draw_statistics(img, stats, x, y):
    """Desenha estatísticas do jogo."""
    cv2.putText(img, "ESTATISTICAS", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    y += 35

    cv2.putText(img, f"Rodadas: {stats.get('total_rounds', 0)}", (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    y += 28

    cv2.putText(img, f"Jogador: {stats.get('player_wins', 0)} ({stats.get('player_win_rate', 0):.1f}%)",
                (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    y += 28

    cv2.putText(img, f"IA: {stats.get('ai_wins', 0)} ({stats.get('ai_win_rate', 0):.1f}%)",
                (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    y += 28

    cv2.putText(img, f"Empates: {stats.get('draws', 0)}",
                (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)


def run_console():
    """Loop principal original, em janela OpenCV."""
    detector = HandDetector()
    game = GameLogic(timer_seconds=3.0, final_window_seconds=0.2)

    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Erro: não foi possível abrir a câmera.")
        return

    print("\n" + "=" * 50)
    print("JOKENPO - CLASSIFICADOR POR REGRAS")
    print("=" * 50)
    print("Comandos: [R] Reiniciar  [Q] Sair")
    print("=" * 50 + "\n")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        detection = detector.encontrar_pontos(frame)
        frame = detection.annotated_frame

        snapshot = game.update(
            hand_visible=detection.visible,
            landmarks=detection.landmarks,
            handedness=detection.handedness,
        )

        state = snapshot["state"]
        timer_text = snapshot["timer_text"]
        player_move = snapshot["player_move"]
        ai_move = snapshot["ai_move"]
        result = snapshot["result"]
        in_final_window = snapshot["in_final_window"]
        stats = snapshot.get("stats", {})

        if state == GameState.WAIT_HAND:
            status_text = "COLOQUE A MAO NA FRENTE DA CAMERA"
            status_color = (0, 0, 255)
        elif state == GameState.COUNTDOWN:
            if in_final_window:
                status_text = f"JANELA FINAL: {timer_text}"
                status_color = (0, 255, 255)
            else:
                status_text = f"CONTAGEM: {timer_text}"
                status_color = (0, 255, 0)
        else:
            status_text = "RESULTADO PRONTO"
            status_color = (255, 255, 0)

        cv2.putText(frame, status_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, status_color, 2)
        cv2.putText(
            frame,
            f"MAO: {'SIM' if detection.visible else 'NAO'} | Mao: {detection.handedness if detection.handedness else 'N/A'}",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
        )

        if state == GameState.COUNTDOWN and detection.visible and detection.landmarks:
            current_cls, current_conf = game.predictor.predict_current(detection.landmarks, handedness=detection.handedness)
            current_text = game.classifier.label_to_text(current_cls)
            color_map = {
                "PEDRA": (100, 100, 255),
                "PAPEL": (100, 255, 100),
                "TESOURA": (255, 100, 100),
                "INDEFINIDO": (200, 200, 200)
            }
            gesture_color = color_map.get(current_text, (255, 255, 255))
            cv2.putText(frame, f"REGRA: {current_text} ({current_conf:.2f})", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, gesture_color, 2)
            samples = (snapshot.get("current_prediction") or {}).get("samples_count", 0)
            cv2.putText(frame, f"Samples: {samples}", (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)

        if state == GameState.RESULT:
            y_pos = 200
            cv2.putText(frame, f"JOGADOR: {player_move}", (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            y_pos += 50
            cv2.putText(frame, f"IA: {ai_move}", (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            y_pos += 60
            if "JOGADOR" in result:
                result_color = (0, 255, 0)
            elif "EMPATE" in result:
                result_color = (200, 200, 200)
            else:
                result_color = (0, 0, 255)
            cv2.putText(frame, f"RESULTADO: {result}", (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 1.0, result_color, 2)
            y_pos += 60
            cv2.putText(frame, "Pressione R para jogar novamente", (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
            draw_statistics(frame, stats, 500, 40)

        cv2.putText(frame, "[R] Reiniciar  [Q] Sair", (20, frame.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        cv2.imshow("Jokenpo - Regras", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("r"):
            game.reset_round()

    cap.release()
    cv2.destroyAllWindows()


def main():
    """Ponto de entrada principal."""
    parser = argparse.ArgumentParser(description="Jokenpo Medieval")
    parser.add_argument(
        "--modo",
        choices=("gui", "console"),
        default="gui",
        help="Escolha a interface de execução. Padrão: gui",
    )
    args = parser.parse_args()

    if args.modo == "console":
        run_console()
        return

    from gui import MedievalJokenpoGUI

    # Instâncias criadas aqui para deixar a GUI orquestrada a partir do mesmo núcleo do jogo.
    detector = HandDetector()
    game = GameLogic(timer_seconds=3.0, final_window_seconds=0.2)
    app = MedievalJokenpoGUI(detector=detector, game=game)
    app.run()


if __name__ == "__main__":
    main()
