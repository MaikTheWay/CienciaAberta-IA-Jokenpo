# =============================================================================
# temporal_predictor.py
# =============================================================================
# Modelo LSTM/GRU para predição temporal de gestos em tempo real.
# Utiliza sequência de landmarks para antecipar a jogada do jogador.
#
# Funcionalidades:
# - Construção de modelo sequencial (LSTM/GRU)
# - Treinamento com dados de landmarks
# - Predição em tempo real com baixa latência
# - Atualização contínua de predições durante a contagem
# - Exportação/carregamento de modelo treinado
#
# Alberto Seleto de Souza / Marcos Alcino Ribeiro Cussioli
# =============================================================================

import os
import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass

# TensorFlow/Keras
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduz logs do TensorFlow

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Scikit-learn para normalização
from sklearn.preprocessing import StandardScaler, MinMaxScaler


@dataclass
class PredictionResult:
    """Resultado de uma predição do modelo temporal."""
    class_id: int           # 0=PEDRA, 1=PAPEL, 2=TESOURA
    class_name: str         # Nome da classe
    confidence: float       # Confiança máxima
    probabilities: np.ndarray  # Array de 3 probabilidades [p_pedra, p_papel, p_tesoura]
    is_stable: bool         # Se a predição é estável
    frame_count: int        # Frames usados na predição


class TemporalModelConfig:
    """Configurações para o modelo temporal."""

    # Dimensões de entrada
    LANDMARK_COUNT = 21
    COORDS_PER_LANDMARK = 3  # x, y, z
    INPUT_FEATURES = LANDMARK_COUNT * COORDS_PER_LANDMARK  # 63

    # Configurações do buffer
    DEFAULT_BUFFER_FRAMES = 12  # ~0.4s a 30fps
    MIN_FRAMES_FOR_PREDICTION = 5

    # Configurações do modelo
    MODEL_TYPE = 'lstm'  # 'lstm' ou 'gru'
    HIDDEN_UNITS = 64
    NUM_LAYERS = 2
    DROPOUT_RATE = 0.3

    # Classes
    CLASSES = ['PEDRA', 'PAPEL', 'TESOURA']
    NUM_CLASSES = 3


class TemporalPredictor:
    """
    Preditor temporal usando modelo LSTM/GRU para antecipação de gestos.

    Este modelo analisa a sequência temporal de landmarks para prever
    qual gesto o jogador está fazendo, permitindo antecipar a jogada
    antes da conclusão do gesto.

    O modelo é executado continuamente durante a contagem regressiva,
    atualizando as probabilidades em tempo real com baixa latência.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        buffer_frames: int = TemporalModelConfig.DEFAULT_BUFFER_FRAMES,
        model_type: str = 'lstm',
        hidden_units: int = TemporalModelConfig.HIDDEN_UNITS,
        num_layers: int = TemporalModelConfig.NUM_LAYERS,
        dropout_rate: float = TemporalModelConfig.DROPOUT_RATE,
        confidence_threshold: float = 0.5,
        stability_window: int = 5,
    ):
        """
        Inicializa o predidor temporal.

        Args:
            model_path: Caminho para modelo pré-treinado (opcional)
            buffer_frames: Número de frames no buffer de sequência
            model_type: 'lstm' ou 'gru'
            hidden_units: Unidades ocultas por camada
            num_layers: Número de camadas RNN
            dropout_rate: Taxa de dropout
            confidence_threshold: Limiar de confiança para predição estável
            stability_window: Janela para verificar estabilidade da predição
        """
        self.buffer_frames = buffer_frames
        self.model_type = model_type.lower()
        self.hidden_units = hidden_units
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.confidence_threshold = confidence_threshold
        self.stability_window = stability_window

        # Modelo Keras
        self.model: Optional[keras.Model] = None
        self.scaler: Optional[StandardScaler] = None

        # Histórico de predições para verificação de estabilidade
        self._prediction_history: List[int] = []
        self._probability_history: List[np.ndarray] = []

        # Carrega modelo existente ou cria novo
        if model_path and os.path.exists(model_path):
            self._load_model(model_path)
        else:
            self.model = self._build_model()
            print(f"[TemporalPredictor] Novo modelo {model_type.upper()} criado")

    def _build_model(self) -> keras.Model:
        """
        Constrói a arquitetura do modelo LSTM/GRU.

        Arquitetura:
        - Input: sequência (batch, timesteps, features)
        - Camadas LSTM/GRU com dropout
        - Dense layers para classificação
        - Output: softmax com 3 classes

        Returns:
            Modelo Keras compilado
        """
        inputs = layers.Input(
            shape=(self.buffer_frames, TemporalModelConfig.INPUT_FEATURES),
            name='input_sequence'
        )

        x = inputs

        # Camadas RNN (LSTM ou GRU)
        for i in range(self.num_layers):
            return_sequences = (i < self.num_layers - 1)

            if self.model_type == 'gru':
                rnn_layer = layers.GRU(
                    self.hidden_units,
                    return_sequences=return_sequences,
                    dropout=self.dropout_rate,
                    recurrent_dropout=0.1,
                    name=f'gru_layer_{i+1}'
                )
            else:  # lstm
                rnn_layer = layers.LSTM(
                    self.hidden_units,
                    return_sequences=return_sequences,
                    dropout=self.dropout_rate,
                    recurrent_dropout=0.1,
                    name=f'lstm_layer_{i+1}'
                )

            x = rnn_layer(x)

        # Camadas densas de classificação
        x = layers.Dense(32, activation='relu', name='dense_1')(x)
        x = layers.Dropout(self.dropout_rate / 2, name='dropout_final')(x)
        outputs = layers.Dense(
            TemporalModelConfig.NUM_CLASSES,
            activation='softmax',
            name='output'
        )(x)

        model = models.Model(inputs=inputs, outputs=outputs, name=f'{self.model_type}_predictor')

        # Compila com Adam e categorical crossentropy
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        return model

    def _load_model(self, model_path: str):
        """Carrega modelo pré-treinado do disco."""
        try:
            self.model = keras.models.load_model(model_path)
            print(f"[TemporalPredictor] Modelo carregado: {model_path}")
        except Exception as e:
            print(f"[TemporalPredictor] Erro ao carregar modelo: {e}")
            print("[TemporalPredictor] Criando novo modelo...")
            self.model = self._build_model()

    def save_model(self, model_path: str):
        """Salva o modelo treinado no disco."""
        if self.model is not None:
            self.model.save(model_path)
            print(f"[TemporalPredictor] Modelo salvo: {model_path}")

    def predict(self, sequence: np.ndarray) -> PredictionResult:
        """
        Faz predição usando uma sequência de landmarks.

        Args:
            sequence: Array numpy shape (frames, 63) com landmarks flatten

        Returns:
            PredictionResult com classe, confiança e probabilidades
        """
        if self.model is None:
            return self._default_result()

        # Verifica quantidade de frames
        if sequence.shape[0] < TemporalModelConfig.MIN_FRAMES_FOR_PREDICTION:
            return self._default_result("frames_insufficient")

        # Padding ou truncamento para tamanho fixo
        if sequence.shape[0] < self.buffer_frames:
            # Padding com último frame
            padding = np.tile(sequence[-1:], (self.buffer_frames - sequence.shape[0], 1))
            sequence = np.vstack([sequence, padding])
        elif sequence.shape[0] > self.buffer_frames:
            # Usa últimos frames
            sequence = sequence[-self.buffer_frames:]

        # Adiciona dimensão de batch
        sequence = np.expand_dims(sequence, axis=0)

        # Predição
        try:
            probabilities = self.model.predict(sequence, verbose=0)[0]
        except Exception as e:
            print(f"[TemporalPredictor] Erro na predição: {e}")
            return self._default_result()

        # Extrai resultado
        class_id = int(np.argmax(probabilities))
        confidence = float(probabilities[class_id])

        # Verifica estabilidade
        is_stable = self._check_stability(class_id, confidence)

        # Atualiza histórico
        self._prediction_history.append(class_id)
        self._probability_history.append(probabilities)

        # Mantém histórico limitado
        if len(self._prediction_history) > self.stability_window * 2:
            self._prediction_history = self._prediction_history[-self.stability_window:]
            self._probability_history = self._probability_history[-self.stability_window:]

        return PredictionResult(
            class_id=class_id,
            class_name=TemporalModelConfig.CLASSES[class_id],
            confidence=confidence,
            probabilities=probabilities,
            is_stable=is_stable,
            frame_count=sequence.shape[1]
        )

    def _check_stability(self, class_id: int, confidence: float) -> bool:
        """
        Verifica se a predição atual é estável.

        Uma predição é considerada estável se:
        1. Confiança >= threshold
        2. A mesma classe apareceu consistentemente na janela recente

        Args:
            class_id: Classe predita
            confidence: Confiança da predição

        Returns:
            True se a predição é estável
        """
        # Confiança deve ser alta
        if confidence < self.confidence_threshold:
            return False

        # Verifica consistência na janela recente
        if len(self._prediction_history) < self.stability_window:
            return False

        recent = self._prediction_history[-self.stability_window:]
        consistency = sum(1 for p in recent if p == class_id) / len(recent)

        return consistency >= 0.7  # 70% de consistência

    def _default_result(self, reason: str = "no_data") -> PredictionResult:
        """Retorna resultado padrão quando não há dados."""
        return PredictionResult(
            class_id=-1,
            class_name="INDEFINIDO",
            confidence=0.0,
            probabilities=np.array([0.0, 0.0, 0.0]),
            is_stable=False,
            frame_count=0
        )

    def get_stabilized_prediction(self) -> PredictionResult:
        """
        Retorna predição estabilizada usando média de probabilidades.

        Útil para reduzir oscilações e ter predição mais suave.

        Returns:
            PredictionResult com predição estabilizada
        """
        if len(self._probability_history) < 3:
            return self._default_result()

        # Média ponderada das últimas probabilidades (mais recentes pesam mais)
        weights = np.linspace(0.5, 1.0, len(self._probability_history))
        weights = weights / weights.sum()

        avg_probs = np.zeros(TemporalModelConfig.NUM_CLASSES)
        for i, probs in enumerate(self._probability_history):
            avg_probs += probs * weights[i]

        # Normaliza
        avg_probs = avg_probs / avg_probs.sum()

        class_id = int(np.argmax(avg_probs))
        confidence = float(avg_probs[class_id])

        return PredictionResult(
            class_id=class_id,
            class_name=TemporalModelConfig.CLASSES[class_id],
            confidence=confidence,
            probabilities=avg_probs,
            is_stable=self._check_stability(class_id, confidence),
            frame_count=len(self._probability_history)
        )

    def reset_history(self):
        """Limpa o histórico de predições."""
        self._prediction_history.clear()
        self._probability_history.clear()

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        epochs: int = 50,
        batch_size: int = 32,
        model_save_path: Optional[str] = None,
        verbose: int = 1
    ) -> Dict:
        """
        Treina o modelo com dados de sequência.

        Args:
            X_train: Array shape (samples, timesteps, features)
            y_train: Array shape (samples, num_classes) ou (samples,) com labels
            X_val: Dados de validação (opcional)
            y_val: Labels de validação (opcional)
            epochs: Número de épocas
            batch_size: Tamanho do batch
            model_save_path: Caminho para salvar melhor modelo
            verbose: Nível de output

        Returns:
            Histórico de treinamento
        """
        if self.model is None:
            print("[TemporalPredictor] Modelo não inicializado!")
            return {}

        # Converte labels para categorical se necessário
        if len(y_train.shape) == 1:
            y_train = keras.utils.to_categorical(y_train, TemporalModelConfig.NUM_CLASSES)

        if y_val is not None and len(y_val.shape) == 1:
            y_val = keras.utils.to_categorical(y_val, TemporalModelConfig.NUM_CLASSES)

        # Callbacks
        callback_list = []

        if model_save_path:
            checkpoint = ModelCheckpoint(
                model_save_path,
                monitor='val_accuracy' if X_val is not None else 'accuracy',
                save_best_only=True,
                mode='max',
                verbose=1
            )
            callback_list.append(checkpoint)

        early_stop = EarlyStopping(
            monitor='val_loss' if X_val is not None else 'loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        )
        callback_list.append(early_stop)

        # Treina
        validation_data = (X_val, y_val) if X_val is not None else None

        history = self.model.fit(
            X_train, y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callback_list,
            verbose=verbose
        )

        return history.history

    def generate_synthetic_data(
        self,
        samples_per_class: int = 200,
        noise_level: float = 0.02
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Gera dados sintéticos para treinamento inicial.

        Cria sequências simuladas representando diferentes gestos
        com variação natural (ruído, velocidade, etc).

        Args:
            samples_per_class: Amostras por classe
            noise_level: Nível de ruído nos landmarks

        Returns:
            Tuple (X, y) com dados de treinamento
        """
        sequences = []
        labels = []

        # Padrões base para cada gesto (21 landmarks simplificados)
        # Representam a posição relativa dos dedos
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
            for _ in range(samples_per_class):
                # Gera sequência de frames
                sequence = []

                for frame_idx in range(self.buffer_frames):
                    # Simula progressão do gesto
                    progress = frame_idx / self.buffer_frames

                    # Calcula estado dos dedos baseado no padrão
                    finger_state = {}
                    for finger, extended in pattern.items():
                        if extended:
                            # Dedo se estende durante a sequência
                            finger_state[finger] = min(1.0, progress * 1.2)
                        else:
                            # Dedo começa estendido e fecha
                            finger_state[finger] = max(0.0, 1.0 - progress * 0.8)

                    # Gera landmarks simulados
                    landmarks = self._generate_frame_landmarks(finger_state, noise_level)
                    sequence.append(landmarks)

                sequences.append(sequence)
                labels.append(class_id)

        X = np.array(sequences, dtype=np.float32)
        y = np.array(labels, dtype=np.int32)

        # Embaralha
        indices = np.random.permutation(len(X))
        X = X[indices]
        y = y[indices]

        return X, y

    def _generate_frame_landmarks(
        self,
        finger_state: Dict[str, float],
        noise_level: float
    ) -> np.ndarray:
        """
        Gera um frame de landmarks simulados.

        Args:
            finger_state: Estado de cada dedo (0=fechado, 1=aberto)
            noise_level: Nível de ruído

        Returns:
            Array de 63 valores (21 landmarks * 3 coords)
        """
        landmarks = []

        # Posição base do pulso
        wrist = np.array([0.5, 0.5, 0.0])

        # Comprimento característico de cada segmento
        finger_lengths = {
            'thumb': 0.08,
            'index': 0.12,
            'middle': 0.13,
            'ring': 0.12,
            'pinky': 0.10
        }

        # Ordem dos dedos no MediaPipe (pulso primeiro)
        finger_order = ['thumb', 'index', 'middle', 'ring', 'pinky']

        for finger_name in finger_order:
            finger_ext = finger_state.get(finger_name, 0.0)
            length = finger_lengths[finger_name]

            # Para cada segmento do dedo
            segments = 4  # CMC/MCP, PIP, DIP, TIP
            prev_pos = wrist.copy()

            for seg in range(segments):
                # Direção base do dedo (para cima)
                base_angle = np.pi / 2 * (1 - finger_ext * 0.8)

                # Adiciona variação por segmento
                angle_variation = np.random.uniform(-0.1, 0.1)
                direction = np.array([
                    np.cos(base_angle + angle_variation) * 0.3,
                    np.sin(base_angle + angle_variation),
                    np.random.uniform(-0.05, 0.05)
                ])

                # Adiciona ruído
                noise = np.random.normal(0, noise_level, 3)
                direction = direction + noise

                # Normaliza e escala
                direction = direction / np.linalg.norm(direction)
                seg_length = length / segments
                pos = prev_pos + direction * seg_length

                # Adiciona à lista de landmarks
                landmarks.extend([pos[0], pos[1], pos[2]])

                prev_pos = pos

        # Preenche para 21 landmarks (7 pontos extras)
        while len(landmarks) < 63:
            noise = np.random.normal(0, noise_level, 3)
            landmarks.extend([0.5 + noise[0], 0.5 + noise[1], noise[2]])

        return np.array(landmarks[:63], dtype=np.float32)

    def summary(self) -> str:
        """Retorna resumo do modelo."""
        if self.model is None:
            return "Modelo não carregado"

        return f"""
=== Temporal Predictor Summary ===
Type: {self.model_type.upper()}
Buffer Frames: {self.buffer_frames}
Input Shape: ({self.buffer_frames}, {TemporalModelConfig.INPUT_FEATURES})
Hidden Units: {self.hidden_units}
Layers: {self.num_layers}
Dropout: {self.dropout_rate}

=== Model Architecture ===
{self.model.summary()}
"""


def create_temporal_predictor(
    model_path: Optional[str] = None,
    buffer_duration: float = 0.4,
    fps: float = 30.0,
    model_type: str = 'lstm'
) -> Tuple[TemporalPredictor, int]:
    """
    Factory function para criar um TemporalPredictor configurado.

    Args:
        model_path: Caminho para modelo pré-treinado
        buffer_duration: Duração do buffer em segundos
        fps: FPS esperado da captura
        model_type: 'lstm' ou 'gru'

    Returns:
        Tuple (predictor, buffer_frames)
    """
    buffer_frames = max(5, int(buffer_duration * fps))

    predictor = TemporalPredictor(
        model_path=model_path,
        buffer_frames=buffer_frames,
        model_type=model_type
    )

    return predictor, buffer_frames
