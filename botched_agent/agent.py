from neural_network import GameNN, evaluate_board
from mcts import mcts_search
from q_learning import get_q_value, update_q_value
from game_logic import get_possible_moves

class Agent:
    def __init__(self, use_mcts=True):
        self.use_mcts = use_mcts
        self.model = GameNN()  # Load or initialize the neural network model
    
    def choose_move(self, board):
        # Neural Network evaluation
        board_value = evaluate_board(board, self.model)
        
        # If MCTS is enabled, use MCTS for decision-making
        if self.use_mcts:
            best_move = mcts_search(board, self.model)
            return best_move
        else:
            # Use Q-learning to choose the best action
            possible_moves = get_possible_moves(board)
            best_action = max(possible_moves, key=lambda action: get_q_value(board, action))
            return best_action
