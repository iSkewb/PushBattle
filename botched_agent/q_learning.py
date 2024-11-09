import numpy as np
from game_logic import get_possible_moves

# Initialize Q-table
Q_table = {}

def get_q_value(state, action):
    return Q_table.get((state, action), 0)

def update_q_value(state, action, reward, next_state, alpha=0.1, gamma=0.9):
    max_future_q = max([get_q_value(next_state, a) for a in get_possible_moves(next_state)], default=0)
    current_q = get_q_value(state, action)
    Q_table[(state, action)] = current_q + alpha * (reward + gamma * max_future_q - current_q)
