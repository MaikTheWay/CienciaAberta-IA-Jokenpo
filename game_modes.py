from enum import Enum, auto
import random
from typing import Dict

class GameMode(Enum):
    DOMINANT_IA = auto()      # IA sempre vence
    PLAYER_FAVORABLE = auto() # Jogador sempre vence
    RANDOM = auto()           # Aleatório

class GameModeManager:
    """
    Gerencia a lógica de decisão da jogada da IA baseada no modo de jogo atual.
    Esta classe encapsula a estratégia de 'trapaça' ou 'justiça' de forma limpa.
    """
    
    def __init__(self):
        # Mapeamento de contra-ataques: o que vence o quê
        self.winning_map = {
            "PEDRA": "PAPEL",
            "PAPEL": "TESOURA",
            "TESOURA": "PEDRA"
        }
        # Mapeamento de derrotas: o que perde para o quê
        self.losing_map = {
            "PEDRA": "TESOURA",
            "PAPEL": "PEDRA",
            "TESOURA": "PAPEL"
        }
        
        # Modo inicial
        self.current_mode = GameMode.RANDOM
        
    def set_mode(self, mode: GameMode):
        """Altera o modo de jogo internamente."""
        self.current_mode = mode

    def get_ai_move(self, player_move: str) -> str:
        """
        Determina a jogada da IA com base no movimento detectado do jogador 
        e no modo de jogo ativo.
        """
        if player_move == "INDEFINIDO":
            return random.choice(["PEDRA", "PAPEL", "TESOURA"])

        if self.current_mode == GameMode.DOMINANT_IA:
            # IA sempre vence: escolhe o que ganha do jogador
            return self.winning_map.get(player_move)
            
        elif self.current_mode == GameMode.PLAYER_FAVORABLE:
            # Jogador sempre vence: IA escolhe o que perde para o jogador
            return self.losing_map.get(player_move)
            
        else:
            # Modo Aleatório: Escolha randômica real
            return random.choice(["PEDRA", "PAPEL", "TESOURA"])

    def update_mode_logic(self, stats: Dict):
        """
        Lógica interna para alternar modos sem intervenção ou aviso visual.
        Pode ser baseada em rodadas, tempo ou probabilidade.
        """
        # Exemplo de lógica: alternar a cada X rodadas ou manter aleatório por padrão
        # Aqui podemos implementar uma lógica que mude o modo silenciosamente.
        # Para este projeto, manteremos uma troca baseada em probabilidade oculta 
        # ou apenas permitiremos que o sistema defina conforme necessário.
        
        # Se quisermos que o modo mude dinamicamente:
        # if stats['total_rounds'] % 5 == 0:
        #     self.current_mode = random.choice(list(GameMode))
        pass
