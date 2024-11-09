import math
import random
from game_logic import get_possible_moves, simulate_move, simulate_game, backpropagate

class MCTSNode:
    def __init__(self, board, parent=None):
        self.board = board
        self.parent = parent
        self.children = []
        self.visits = 0
        self.wins = 0
        self.untried_actions = get_possible_moves(board)
    
    def uct_value(self, c=1.4):
        if self.visits == 0:
            return float('inf')
        return self.wins / self.visits + c * math.sqrt(math.log(self.parent.visits) / self.visits)

def mcts_search(board, model, num_simulations=1000):
    root = MCTSNode(board)
    
    for _ in range(num_simulations):
        node = root
        while node.untried_actions == [] and node.children:
            node = max(node.children, key=lambda child: child.uct_value())
        if node.untried_actions:
            move = node.untried_actions.pop()
            new_board = simulate_move(node.board, move)
            child = MCTSNode(new_board, parent=node)
            node.children.append(child)
        else:
            simulate_game(node.board, model)
        
        # Backpropagate the results
        backpropagate(node, model)
    
    return max(root.children, key=lambda child: child.visits).board
