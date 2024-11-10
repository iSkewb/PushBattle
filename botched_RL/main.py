import os
from agent import RLAgent
from PushBattle import Game, PLAYER1, PLAYER2
from PushBattle import BOARD_SIZE
import numpy as np

def select_empty_position(game_board):
    # This function should return the row and column indices of an empty space
    # Loop through the game board and return the coordinates of the first empty position found
    for r in range(len(game_board)):
        for c in range(len(game_board[r])):
            if game_board[r][c] == 0:  # Assuming 0 means empty space
                return r, c
    return None, None  # Return None if no empty space is found

def train_agent(episodes=1000, save_interval=100, model_path="agent.pth"):
    player1 = RLAgent(player=PLAYER1)
    player2 = RLAgent(player=PLAYER2)

    if os.path.exists(model_path):
        print("Loading existing model...")
        player1.load(model_path)

    for episode in range(episodes):
        game = Game()
        state = player1.get_state(game)
        done = False
        
        # Print initial state shape to verify its structure
        print("Initial state shape:", state.shape)
        
        while not done:
            # Check if state is a dictionary (and if so, flatten it)
            if isinstance(state, dict):
                # Flatten the dictionary to a numpy array
                state = np.concatenate([np.array(value).flatten() for value in state.values()])
            
            # Print flattened state shape
            print("Flattened state shape:", state.shape)

            # Perform action and get next state
            action1 = player1.get_action(state)
            action1 = format_action(action1)  # Convert to correct format
            next_state, reward, done = game.step(action1)

            # Ensure next_state is a numpy array (only flatten if it's a dictionary)
            if isinstance(next_state, dict):
                # Flatten the dictionary to a numpy array
                next_state = np.concatenate([np.array(value).flatten() for value in next_state.values()])
            
            # Print next_state shape after flattening (if applicable)
            print("Next state shape:", next_state.shape)
            next_state = next_state[:64]  # Trim to 64 dimensions if it grows beyond this
            print("Trimmed next_state shape:", next_state.shape)

            # Store experience in player1's memory
            player1.store_experience(state, action1, reward, next_state, done)
            state = next_state

            # Handle player2's action if not done
            if not done:
                action2 = player2.get_action(state)
                action2 = format_action(action2)  # Convert to correct format
                next_state, reward, done = game.step(action2)

                # Ensure next_state is a numpy array (only flatten if it's a dictionary)
                if isinstance(next_state, dict):
                    next_state = np.concatenate([np.array(value).flatten() for value in next_state.values()])
                
                # Print next_state shape after flattening (if applicable)
                print("Next state (after player2 action) shape:", next_state.shape)

                # Ensure next_state size remains constant
                next_state = next_state[:64]  # Trim to 64 dimensions if it grows beyond this
                print("Trimmed next_state shape:", next_state.shape)


                # Store experience in player2's memory
                player2.store_experience(state, action2, -reward, next_state, done)
                state = next_state

            # Replay experience for player1 and player2
            player1.replay()
            player2.replay()

        # Periodically save the model
        if episode % save_interval == 0:
            print(f"Episode {episode}/{episodes}, epsilon: {player1.epsilon:.2f}")
            player1.save(model_path)

    # Save final model
    player1.save(model_path)



# A possible implementation of format_action function
def format_action(action):
    """Converts an action into a tuple or a list for step() method."""
    if isinstance(action, int):
        # Assuming action is an integer representing the position on the board
        r, c = divmod(action, BOARD_SIZE)  # Convert action to row and column
        return ('place', r, c)  # Return action in tuple format
    # Add other conditions for different types of actions, like 'move'
    else:
        raise ValueError("Unsupported action type.")



if __name__ == "__main__":
    train_agent(episodes=5000)
