# DOC;
# Orquestra tudo: inicializa câmera, detector, classificador, timer, preditor.
# Executa o loop infinito até que o usuário saia.

# Alberto Seleto de Souza
# Marcos Alcino Ribeiro Cussioli

import cv2
from hand_detector import HandDetector
from game_logic import GameLogic, GameState


def main():
    detector = HandDetector()
    game = GameLogic(timer_seconds=3.0, final_window_seconds=0.5)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        cap = cv2.VideoCapture(1)

    if not cap.isOpened():
        print("Erro: não foi possível abrir a câmera.")
        return

    print("--- SISTEMA DE JOKENPO IA ---")
    print("Comandos: [R] reiniciar | [Q] sair")

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

        if state == GameState.WAIT_HAND:
            status_text = "COLOQUE A MAO NA FRENTE DA CAMERA"
            status_color = (0, 0, 255)
        elif state == GameState.COUNTDOWN:
            if in_final_window:
                status_text = f"JANELA FINAL: {timer_text}"
            else:
                status_text = f"CONTAGEM: {timer_text}"
            status_color = (0, 255, 0)
        else:
            status_text = "RESULTADO PRONTO"
            status_color = (0, 255, 255)

        cv2.putText(
            frame,
            status_text,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            status_color,
            2,
        )

        cv2.putText(
            frame,
            f"MAO: {detection.visible} | HAND: {detection.handedness if detection.handedness else 'N/A'}",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        if state == GameState.COUNTDOWN and detection.visible and detection.landmarks:
            current_cls, current_conf = game.predictor.predict_current(
                detection.landmarks,
                handedness=detection.handedness,
            )
            current_text = game.classifier.label_to_text(current_cls)
            cv2.putText(
                frame,
                f"LEITURA ATUAL: {current_text} ({current_conf:.2f})",
                (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

        if state == GameState.RESULT:
            cv2.putText(
                frame,
                f"JOGADOR: {player_move}",
                (20, 170),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                frame,
                f"IA: {ai_move}",
                (20, 210),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                frame,
                f"RESULTADO: {result}",
                (20, 260),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 255),
                2,
            )
            cv2.putText(
                frame,
                "Pressione R para jogar novamente",
                (20, 300),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (200, 200, 200),
                2,
            )

        cv2.imshow("Jokenpo IA", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        if key == ord("r"):
            game.reset_round()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()