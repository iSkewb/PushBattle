# agent.py
import random
import numpy as np
import torch
from collections import deque
from model import DQNetwork
from config import *

class RLAgent:
    def __init__(self, player, input_size=BOARD_SIZE * BOARD_SIZE, output_size=BOARD_SIZE * BOARD_SIZE):
        self.player = player
        self.epsilon = EPSILON
        self.memory = deque(maxlen=MEMORY_SIZE)
        self.model = DQNetwork(input_size, output_size)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        self.criterion = torch.nn.MSELoss()

    def get_state(self, game):
        state = np.array(game.board).flatten()
        # Ensure the state remains the same size
        # state = state[:64]  # Force state size to be 64 if it is unexpectedly larger
        return state

    def get_action(self, state):
        if np.random.rand() < self.epsilon:
            return random.randint(0, BOARD_SIZE * BOARD_SIZE - 1)
        else:
            state = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                q_values = self.model(state)
            return torch.argmax(q_values).item()

    def store_experience(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        if len(self.memory) < BATCH_SIZE:
            return
        batch = random.sample(self.memory, BATCH_SIZE)
        for state, action, reward, next_state, done in batch:
            state = torch.FloatTensor(state)
            next_state = torch.FloatTensor(next_state)
            target = reward
            if not done:
                target += GAMMA * torch.max(self.model(next_state)).item()
            target_f = self.model(state)
            target_f[action] = target
            loss = self.criterion(target_f, torch.FloatTensor([target]))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        if self.epsilon > EPSILON_MIN:
            self.epsilon *= EPSILON_DECAY

    def save(self, path="agent.pth"):
        self.model.save_model(path)

    def load(self, path="agent.pth"):
        self.model = DQNetwork.load_model(path, input_size=BOARD_SIZE * BOARD_SIZE, output_size=BOARD_SIZE * BOARD_SIZE)

    def format_action(self, action):
        # Assuming action is an integer, we need to map it into a valid action format
        r0, c0 = divmod(action, BOARD_SIZE)  # Just an example conversion logic
        action_type = 'move'  # or 'place' depending on the action type
        return (action_type, r0, c0)  # Adjust as needed
