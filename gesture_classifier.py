# DOC;
# Recebe os landmarks da mão e decide se é pedra, papel ou tesoura.
# Abordagem inicial: regras heurísticas com base na distância entre dedos.
# Abordagem avançada: pequeno modelo de aprendizado (ex: MLP) treinado com dados coletados.

import os
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout

class BrainJokenpo:
    def __init__(self, model_path='training_model/modelo_jokenpo.h5', dataset_path='training_model/dados_jokenpo.csv'):
        self.model_path = model_path
        self.dataset_path = dataset_path
        self.input_size = 63 # 21 pontos * (x,y,z)
        
        if os.path.exists(model_path):
            self.model = load_model(model_path)
        else:
            self.model = self._criar_modelo_vazio()

    def _criar_modelo_vazio(self):
        model = Sequential([
            Dense(128, activation='relu', input_shape=(self.input_size,)),
            Dropout(0.2),
            Dense(64, activation='relu'),
            Dense(3, activation='softmax') # [Pedra, Papel, Tesoura]
        ])
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def salvar_e_treinar_ponto(self, lista_pontos, label):
        # Salva no CSV para persistência
        df = pd.DataFrame([lista_pontos + [label]])
        df.to_csv(self.dataset_path, mode='a', header=not os.path.exists(self.dataset_path), index=False)
        
        # Treino rápido (incremental)
        X = np.array([lista_pontos])
        y = np.zeros((1, 3))
        y[0, label] = 1
        self.model.fit(X, y, epochs=5, verbose=0)
        self.model.save(self.model_path)

    def prever(self, lista_pontos):
        X = np.array([lista_pontos])
        predicao = self.model.predict(X, verbose=0)
        classe = np.argmax(predicao)
        confianca = np.max(predicao)
        return classe, confianca
