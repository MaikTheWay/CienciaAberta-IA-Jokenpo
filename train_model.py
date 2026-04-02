# =============================================================================
# train_model.py
# =============================================================================
# Script para treinar o modelo LSTM/GRU de predição temporal.
# Gera dados sintéticos ou pode ser expandido para usar dados reais.
#
# Uso:
#   python train_model.py --samples 500 --epochs 50 --model lstm
#   python train_model.py --samples 1000 --epochs 100 --model gru
#
# Alberto Seleto de Souza / Marcos Alcino Ribeiro Cussioli
# =============================================================================

import argparse
import numpy as np
import os
import sys
from datetime import datetime

# TensorFlow - reduz logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.keras.utils import plot_model

from temporal_predictor import TemporalPredictor, create_temporal_predictor, TemporalModelConfig


def generate_enhanced_data(
    samples_per_class: int = 500,
    buffer_frames: int = 12,
    noise_level: float = 0.025,
    augmentation_factor: int = 3
) -> tuple:
    """
    Gera dados de treinamento com variação e augmentation.

    Args:
        samples_per_class: Amostras base por classe
        buffer_frames: Frames por sequência
        noise_level: Nível de ruído nos landmarks
        augmentation_factor: Fator de augmentation

    Returns:
        Tuple (X_train, y_train, X_val, y_val)
    """
    print(f"[Train] Gerando dados: {samples_per_class} samples/classe x {augmentation_factor} augmentations")

    all_sequences = []
    all_labels = []

    # Padrões base para cada gesto
    base_patterns = {
        0: {  # PEDRA - mão fechada
            'thumb_ext': 0,
            'index_ext': 0,
            'middle_ext': 0,
            'ring_ext': 0,
            'pinky_ext': 0
        },
        1: {  # PAPEL - mão aberta
            'thumb_ext': 1,
            'index_ext': 1,
            'middle_ext': 1,
            'ring_ext': 1,
            'pinky_ext': 1
        },
        2: {  # TESOURA - indicador e médio estendidos
            'thumb_ext': 0,
            'index_ext': 1,
            'middle_ext': 1,
            'ring_ext': 0,
            'pinky_ext': 0
        }
    }

    for class_id, pattern in base_patterns.items():
        for aug in range(augmentation_factor):
            # Variação de parâmetros por augmentation
            aug_noise = noise_level * (1 + aug * 0.3)
            aug_speed = 0.8 + aug * 0.15  # Variação de velocidade do gesto

            for sample_idx in range(samples_per_class):
                sequence = []

                for frame_idx in range(buffer_frames):
                    # Progressão do gesto
                    progress = (frame_idx / buffer_frames) * aug_speed
                    progress = min(1.0, progress)

                    # Estado dos dedos
                    finger_state = {}
                    for finger, extended in pattern.items():
                        if extended:
                            finger_state[finger] = min(1.0, progress * 1.3)
                        else:
                            finger_state[finger] = max(0.0, 1.0 - progress * 0.9)

                    # Gera landmarks
                    landmarks = _generate_frame_landmarks(finger_state, aug_noise)
                    sequence.append(landmarks)

                all_sequences.append(sequence)
                all_labels.append(class_id)

    # Converte para arrays
    X = np.array(all_sequences, dtype=np.float32)
    y = np.array(all_labels, dtype=np.int32)

    # Embaralha
    indices = np.random.permutation(len(X))
    X = X[indices]
    y = y[indices]

    # Split 80/20
    split_idx = int(len(X) * 0.8)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    print(f"[Train] Treino: {len(X_train)} | Validação: {len(X_val)}")

    return X_train, y_train, X_val, y_val


def _generate_frame_landmarks(
    finger_state: dict,
    noise_level: float
) -> np.ndarray:
    """Gera um frame de landmarks simulados."""
    landmarks = []

    wrist = np.array([0.5, 0.5, 0.0])

    finger_lengths = {
        'thumb': 0.08,
        'index': 0.12,
        'middle': 0.13,
        'ring': 0.12,
        'pinky': 0.10
    }

    finger_order = ['thumb', 'index', 'middle', 'ring', 'pinky']

    for finger_name in finger_order:
        finger_ext = finger_state.get(finger_name, 0.0)
        length = finger_lengths[finger_name]

        prev_pos = wrist.copy()

        for seg in range(4):
            base_angle = np.pi / 2 * (1 - finger_ext * 0.8)
            angle_variation = np.random.uniform(-0.15, 0.15)

            direction = np.array([
                np.cos(base_angle + angle_variation) * 0.3,
                np.sin(base_angle + angle_variation),
                np.random.uniform(-0.05, 0.05)
            ])

            noise = np.random.normal(0, noise_level, 3)
            direction = direction + noise

            direction = direction / np.linalg.norm(direction)
            seg_length = length / 4
            pos = prev_pos + direction * seg_length

            landmarks.extend([pos[0], pos[1], pos[2]])
            prev_pos = pos

    while len(landmarks) < 63:
        noise = np.random.normal(0, noise_level, 3)
        landmarks.extend([0.5 + noise[0], 0.5 + noise[1], noise[2]])

    return np.array(landmarks[:63], dtype=np.float32)


def train_model(
    model_type: str = 'lstm',
    samples_per_class: int = 500,
    epochs: int = 50,
    batch_size: int = 32,
    buffer_frames: int = 12,
    output_path: str = 'training_model/temporal_rps_model.h5',
    save_plots: bool = True
):
    """
    Treina o modelo de predição temporal.

    Args:
        model_type: 'lstm' ou 'gru'
        samples_per_class: Amostras por classe
        epochs: Número de épocas
        batch_size: Tamanho do batch
        buffer_frames: Frames no buffer
        output_path: Caminho para salvar modelo
        save_plots: Se deve salvar gráficos de treinamento
    """
    print("\n" + "=" * 60)
    print(f"TREINAMENTO DO MODELO TEMPORAL ({model_type.upper()})")
    print("=" * 60)

    # Cria diretório de output
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)

    # Gera dados
    print("\n[1/4] Gerando dados de treinamento...")
    X_train, y_train, X_val, y_val = generate_enhanced_data(
        samples_per_class=samples_per_class,
        buffer_frames=buffer_frames,
        augmentation_factor=3
    )

    # Cria modelo
    print(f"\n[2/4] Criando modelo {model_type.upper()}...")
    predictor = TemporalPredictor(
        buffer_frames=buffer_frames,
        model_type=model_type,
        hidden_units=64,
        num_layers=2,
        dropout_rate=0.3
    )

    print(f"\nArquitetura do modelo:")
    print(f"  - Tipo: {model_type.upper()}")
    print(f"  - Input: ({buffer_frames}, 63)")
    print(f"  - Hidden units: 64")
    print(f"  - Layers: 2")
    print(f"  - Output: 3 classes")

    # Treina
    print(f"\n[3/4] Treinando ({epochs} épocas, batch_size={batch_size})...")

    history = predictor.train(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        epochs=epochs,
        batch_size=batch_size,
        model_save_path=output_path,
        verbose=1
    )

    # Salva modelo final
    predictor.save_model(output_path)
    print(f"\n[4/4] Modelo salvo em: {output_path}")

    # Métricas finais
    if history:
        print("\n" + "-" * 40)
        print("RESULTADOS DO TREINAMENTO")
        print("-" * 40)

        final_epoch = len(history.get('accuracy', [0]))
        print(f"Épocas treinadas: {final_epoch}")

        train_acc = history.get('accuracy', [0])[-1]
        val_acc = history.get('val_accuracy', [0])[-1] if history.get('val_accuracy') else train_acc
        train_loss = history.get('loss', [0])[-1]
        val_loss = history.get('val_loss', [0])[-1] if history.get('val_loss') else train_loss

        print(f"\nAcurácia final:")
        print(f"  - Treino: {train_acc*100:.2f}%")
        print(f"  - Validação: {val_acc*100:.2f}%")

        print(f"\nLoss final:")
        print(f"  - Treino: {train_loss:.4f}")
        print(f"  - Validação: {val_loss:.4f}")

    # Salva histórico
    if history:
        history_path = output_path.replace('.h5', '_history.npy')
        np.save(history_path, history)
        print(f"\nHistórico salvo em: {history_path}")

    print("\n" + "=" * 60)
    print("TREINAMENTO CONCLUÍDO!")
    print("=" * 60)

    return predictor, history


def main():
    parser = argparse.ArgumentParser(
        description='Treina modelo LSTM/GRU para predição temporal de gestos'
    )

    parser.add_argument(
        '--model', '-m',
        type=str,
        default='lstm',
        choices=['lstm', 'gru'],
        help='Tipo de modelo (default: lstm)'
    )

    parser.add_argument(
        '--samples', '-s',
        type=int,
        default=500,
        help='Amostras por classe (default: 500)'
    )

    parser.add_argument(
        '--epochs', '-e',
        type=int,
        default=50,
        help='Número de épocas (default: 50)'
    )

    parser.add_argument(
        '--batch', '-b',
        type=int,
        default=32,
        help='Batch size (default: 32)'
    )

    parser.add_argument(
        '--buffer', '-bf',
        type=int,
        default=12,
        help='Frames no buffer (default: 12)'
    )

    parser.add_argument(
        '--output', '-o',
        type=str,
        default='training_model/temporal_rps_model.h5',
        help='Caminho de saída do modelo'
    )

    parser.add_argument(
        '--no-save',
        action='store_true',
        help='Não salvar modelo após treinamento'
    )

    args = parser.parse_args()

    # Treina
    predictor, history = train_model(
        model_type=args.model,
        samples_per_class=args.samples,
        epochs=args.epochs,
        batch_size=args.batch,
        buffer_frames=args.buffer,
        output_path=args.output if not args.no_save else None
    )

    if not args.no_save:
        print(f"\nModelo pronto para uso em: {args.output}")
        print("Para usar, passe o caminho ao inicializar o Predictor:")
        print(f"  predictor = Predictor(..., temporal_model_path='{args.output}')")


if __name__ == '__main__':
    main()
