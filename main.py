# DOC;
# Orquestra tudo: inicializa câmera, detector, classificador, timer, preditor.
# Executa o loop infinito até que o usuário saia.

# Alberto Seleto de Souza
# Marcos Alcino Ribeiro Cussioli

import cv2
import numpy as np
from hand_detector import HandDetector
from gesture_classifier import BrainJokenpo

def main():
    detector = HandDetector()
    ia = BrainJokenpo()
    cap = cv2.VideoCapture(1)
    
    print("--- SISTEMA DE JOKENPO IA ---")
    print("Comandos: [1] Treinar Pedra | [2] Treinar Papel | [3] Treinar Tesoura | [Q] Sair")

    while cap.isOpened():
        success, frame = cap.read()
        if not success: break
        
        frame = cv2.flip(frame, 1)
        
        # 1. Extração de Coordenadas (Passagem de variáveis)
        lista_pontos, frame = detector.encontrar_pontos(frame)
        
        mao_visivel = len(lista_pontos) > 0
        gesto_predito = "Nenhum"
        confianca = 0.0

        if mao_visivel:
            # 2. Previsão em Tempo Real
            classe, conf = ia.prever(lista_pontos)
            labels = ["PEDRA", "PAPEL", "TESOURA"]
            gesto_predito = labels[classe]
            confianca = conf

        # --- INTERFACE ---
        cor_status = (0, 255, 0) if mao_visivel else (0, 0, 255)
        cv2.putText(frame, f"MAO: {mao_visivel} | GESTO: {gesto_predito}", (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, cor_status, 2)
        cv2.putText(frame, f"CONFIANCA: {confianca:.2f}", (20, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        cv2.imshow('Jokenpo IA - Main', frame)
        key = cv2.waitKey(1) & 0xFF
        
        # 3. Lógica de Treinamento Constante (Captura Manual para Aprendizado)
        if mao_visivel:
            if key == ord('1'):
                ia.salvar_e_treinar_ponto(lista_pontos, 0)
                print("Aprendendo PEDRA...")
            elif key == ord('2'):
                ia.salvar_e_treinar_ponto(lista_pontos, 1)
                print("Aprendendo PAPEL...")
            elif key == ord('3'):
                ia.salvar_e_treinar_ponto(lista_pontos, 2)
                print("Aprendendo TESOURA...")
        
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
