import numpy as np

# GLOBAL VARIABLES
EMPTY = 0       # Empty space board value
PLAYER1 = 1     # First player board value
PLAYER2 = -1    # Second player board value

BOARD_SIZE = 8  # Size of the board
NUM_PIECES = 8  # Number of pieces each player is allowed to place

##################

def _torus(r, c):
    rt = (r + BOARD_SIZE) % BOARD_SIZE
    ct = (c + BOARD_SIZE) % BOARD_SIZE
    return rt, ct

def array_to_chess_notation(move):
    def to_notation(row, col):
        return f"{chr(ord('a') + col)}{8 - row}"
    return to_notation(move[0], move[1]) + (to_notation(move[2], move[3]) if len(move) == 4 else "")

def chess_notation_to_array(notation):
    def to_array(pos):
        return [8 - int(pos[1]), ord(pos[0]) - ord('a')]
    return to_array(notation[:2]) + (to_array(notation[2:]) if len(notation) == 4 else [])

class Game:
    def __init__(self):
        self.board = np.full((BOARD_SIZE, BOARD_SIZE), EMPTY)
        self.current_player = PLAYER1
        self.turn_count = 0
        self.p1_pieces = 0
        self.p2_pieces = 0

    def to_dict(self):
        return {
            "board": self.board.tolist(),
            "current_player": self.current_player,
            "turn_count": self.turn_count,
            "p1_pieces": self.p1_pieces,
            "p2_pieces": self.p2_pieces,
        }
    
    @classmethod
    def from_dict(cls, data):
        game = cls()
        game.board = np.array(data["board"])
        game.current_player = data["current_player"]
        game.turn_count = data["turn_count"]
        game.p1_pieces = data["p1_pieces"]
        game.p2_pieces = data["p2_pieces"]
        return game

    def display_board(self):
        tile_symbols = {EMPTY: '.', PLAYER2: 'B', PLAYER1: 'W'}
        for row in self.board:
            print(' '.join(tile_symbols[tile] for tile in row))

    def is_valid_placement(self, row, col):
        if self.current_player == PLAYER1 and self.p1_pieces >= NUM_PIECES:
            return False
        if self.current_player == PLAYER2 and self.p2_pieces >= NUM_PIECES:
            return False
        return 0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE and self.board[row][col] == EMPTY

    def is_valid_move(self, r0, c0, r1, c1):
        if not (0 <= r0 < BOARD_SIZE and 0 <= c0 < BOARD_SIZE and 
                0 <= r1 < BOARD_SIZE and 0 <= c1 < BOARD_SIZE):
            return False
        if self.board[r0][c0] != self.current_player:
            return False
        if self.board[r1][c1] != EMPTY:
            return False
        return True
     
    def place_checker(self, r, c):
        self.board[r][c] = self.current_player
        if self.current_player == PLAYER1:
            self.p1_pieces += 1
        else:
            self.p2_pieces += 1
        self.push_neighbors(r, c)

    def move_checker(self, r0, c0, r1, c1):
        self.board[r0][c0] = EMPTY
        self.board[r1][c1] = self.current_player
        self.push_neighbors(r1, c1)

    def push_neighbors(self, r0, c0):
        dirs = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]
        for dr, dc in dirs:
            r1, c1 = _torus(r0 + dr, c0 + dc)
            if self.board[r1][c1] != EMPTY:
                r2, c2 = _torus(r1 + dr, c1 + dc)
                if self.board[r2][c2] == EMPTY:
                    self.board[r2][c2], self.board[r1][c1] = self.board[r1][c1], self.board[r2][c2]

    def check_winner(self):
        player1_wins = False
        player2_wins = False
        for row in range(BOARD_SIZE):
            cnt, tile = 0, EMPTY
            for col in range(-2, BOARD_SIZE+2):
                r, c = _torus(row, col)
                curr_tile = self.board[r][c]
                cnt = cnt + 1 if curr_tile == tile and curr_tile != EMPTY else 1
                tile = curr_tile
                if cnt == 3:
                    if tile == PLAYER1:
                        player1_wins = True
                    elif tile == PLAYER2:
                        player2_wins = True
        for col in range(BOARD_SIZE):
            cnt, tile = 0, EMPTY
            for row in range(-2, BOARD_SIZE+2):
                r, c = _torus(row, col)
                curr_tile = self.board[r][c]
                cnt = cnt + 1 if curr_tile == tile and curr_tile != EMPTY else 1
                tile = curr_tile
                if cnt == 3:
                    if tile == PLAYER1:
                        player1_wins = True
                    elif tile == PLAYER2:
                        player2_wins = True
        for col_start in range(BOARD_SIZE):
            cnt, tile = 0, EMPTY
            for i in range(-2, BOARD_SIZE+2):
                r, c = _torus(i, col_start + i)
                curr_tile = self.board[r][c]
                cnt = cnt + 1 if curr_tile == tile and curr_tile != EMPTY else 1
                tile = curr_tile
                if cnt == 3:
                    if tile == PLAYER1:
                        player1_wins = True
                    elif tile == PLAYER2:
                        player2_wins = True
        for col_start in range(BOARD_SIZE):
            cnt, tile = 0, EMPTY
            for i in range(-2, BOARD_SIZE+2):
                r, c = _torus(i, col_start - i)
                curr_tile = self.board[r][c]
                cnt = cnt + 1 if curr_tile == tile and curr_tile != EMPTY else 1
                tile = curr_tile
                if cnt == 3:
                    if tile == PLAYER1:
                        player1_wins = True
                    elif tile == PLAYER2:
                        player2_wins = True
        if player1_wins and player2_wins:
            return self.current_player
        elif player1_wins:
            return PLAYER1
        elif player2_wins:
            return PLAYER2
        return EMPTY

    def step(self, action):
        # Check if action is a tuple or list and its first element is a string
        if not isinstance(action, (list, tuple)) or not isinstance(action[0], str):
            raise ValueError("Action must be a tuple or list, with a string as the first element to specify action type.")
        
        action_type = action[0]
        reward, done = 0, False
        
        # Logic for placing a checker (action_type == 'place')
        if action_type == 'place':
            row, col = action[1], action[2]
            if self.is_valid_placement(row, col):
                self.place_checker(row, col)
                reward = 1  # Reward for a valid placement
            else:
                reward = -1  # Penalty for invalid placement
        
        # Logic for moving a checker (action_type == 'move')
        elif action_type == 'move':
            r0, c0, r1, c1 = action[1], action[2], action[3], action[4]
            if self.is_valid_move(r0, c0, r1, c1):
                self.move_checker(r0, c0, r1, c1)
                reward = 1  # Reward for a valid move
            else:
                reward = -1  # Penalty for invalid move
        
        # Increment turn count
        self.turn_count += 1

        # Check if the game is over
        winner = self.check_winner()
        if winner != EMPTY:
            done = True  # Game is over
            reward = 10 if winner == self.current_player else -10  # Positive reward for win, negative for loss

        # Switch to the next player
        self.current_player = PLAYER2 if self.current_player == PLAYER1 else PLAYER1

        # Get the state before the action
        print("State before action:", self.get_state().shape)

        # Get the next state
        next_state = self.get_state()

        # Print the state after the action
        print("State after action:", next_state.shape)

        return next_state, reward, done

    def get_state(self):
        return np.append(self.board.flatten(), self.current_player)

def main():
    game = Game()
    game.play()

if __name__ == '__main__':
    main()
