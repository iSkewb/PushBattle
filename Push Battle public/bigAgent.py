import random
import math
from PushBattle import Game, PLAYER1, PLAYER2, EMPTY, BOARD_SIZE, NUM_PIECES
import time

class ImprovedAgent:
    """
    ImprovedAgent class for the Push Battle game.
    This agent uses Monte Carlo Tree Search (MCTS) and heuristic evaluation to determine the best moves in the game.
    Attributes:
        player (int): The player number (PLAYER1 or PLAYER2).
        simulations (int): Number of simulations for MCTS.
        center (tuple): The center position of the board.
    Methods:
        __init__(player=PLAYER1, simulations=50):
            Initializes the agent with the specified player and number of simulations.
        get_possible_moves(game):
            Returns a list of all valid moves in the current state.
        is_valid_placement_move(game, move):
            Validates a placement move.
        is_valid_movement_move(game, move):
            Validates a movement move.
        simulate_game(game, move):
            Simulates a random play-out after the given move.
        apply_move_if_valid(game, move):
            Applies a move if it is valid, returning True if successful.
        mcts(game):
            Uses MCTS to choose the best move.
        evaluate_move(game, move):
            Heuristic to assign a score to each move.
        is_edge_position(r, c):
            Checks if a position is at the edge of the board.
        get_best_move(game):
            Chooses the move based on MCTS if possible, otherwise falls back to heuristic.
    """
    def __init__(self, player=PLAYER1, simulations=50):
        self.player = player
        self.time_limit = 4.5
        self.simulations = simulations  # Number of simulations for MCTS
        self.center = (BOARD_SIZE // 2, BOARD_SIZE // 2)

    def get_possible_moves(self, game):
        """Returns list of all valid moves in the current state."""
        moves = []
        current_pieces = game.p1_pieces if game.current_player == PLAYER1 else game.p2_pieces

        if current_pieces < NUM_PIECES:
            # Only allow placement moves if all pieces have not yet been placed
            for r in range(BOARD_SIZE):
                for c in range(BOARD_SIZE):
                    if game.board[r][c] == EMPTY and self.is_valid_placement_move(game, (r, c)):
                        moves.append((r, c))
        else:
            # Allow movement moves only once all pieces have been placed
            for r0 in range(BOARD_SIZE):
                for c0 in range(BOARD_SIZE):
                    if game.board[r0][c0] == game.current_player:
                        for r1 in range(BOARD_SIZE):
                            for c1 in range(BOARD_SIZE):
                                if game.board[r1][c1] == EMPTY and self.is_valid_movement_move(game, (r0, c0, r1, c1)):
                                    moves.append((r0, c0, r1, c1))
        return moves


    def is_valid_placement_move(self, game, move):
        """Validates a placement move."""
        r, c = move
        # Check if the position is empty
        if game.board[r][c] != EMPTY:
            return False
        # Additional checks based on game rules can be added here
        return True

    def is_valid_movement_move(self, game, move):
        """Validates a movement move."""
        r0, c0, r1, c1 = move
        # Check if the start position has the current player's piece
        if game.board[r0][c0] != game.current_player:
            return False
        # Check if the destination position is empty
        if game.board[r1][c1] != EMPTY:
            return False
        # Check if the move is within bounds
        if not (0 <= r1 < BOARD_SIZE and 0 <= c1 < BOARD_SIZE):
            return False
        # Check if the move is a valid push (e.g., adjacent and in a straight line)
        if abs(r0 - r1) + abs(c0 - c1) != 1:
            return False
        # Additional checks based on game rules can be added here
        return True

    def simulate_game(self, game, move):
        """Simulates a random play-out after the given move."""
        game_copy = Game()
        game_copy.board = [row[:] for row in game.board]
        game_copy.current_player = game.current_player
        game_copy.p1_pieces = game.p1_pieces
        game_copy.p2_pieces = game.p2_pieces
        if not self.apply_move_if_valid(game_copy, move):
            return -1  # Invalid moves should be penalized in scoring

        # Play randomly until the game ends
        while not game_copy.is_game_over():
            possible_moves = self.get_possible_moves(game_copy)
            if not possible_moves:
                break
            random_move = random.choice(possible_moves)
            self.apply_move_if_valid(game_copy, random_move)
        
        # Return the outcome of the game
        return 1 if game_copy.winner() == self.player else -1

    def apply_move_if_valid(self, game, move):
        """Applies a move if it is valid, returning True if successful."""
        possible_moves = self.get_possible_moves(game)
        # Ensure the format matches expected move format for `apply_move`
        if move in possible_moves:
            game.apply_move(move)
            return True
        else:
            print(f"Invalid move format: {move} not in {possible_moves}")
        return False


    def mcts(self, game):
        """Uses MCTS to choose the best move within the time limit."""
        possible_moves = self.get_possible_moves(game)
        if not possible_moves:
            return None  # No valid moves

        move_scores = {move: 0 for move in possible_moves}

        start_time = time.time()
        num_simulations = 0

        # Run MCTS simulations until the time limit is reached
        while time.time() - start_time < self.time_limit:
            for move in possible_moves:
                outcome = self.simulate_game(game, move)
                move_scores[move] += outcome
                num_simulations += 1

                # Check if the time limit is reached within this inner loop
                if time.time() - start_time >= self.time_limit:
                    break
        
        # Print the number of simulations for debugging or tuning purposes
        print(f"Simulations completed: {num_simulations}")

        # Choose the move with the highest average score
        best_move = max(move_scores, key=move_scores.get)
        return best_move

    def evaluate_move(self, game, move):
        """Heuristic to assign a score to each move."""
        if len(move) == 2:  # Placement move
            r, c = move
        else:  # Movement move
            r, c = move[2], move[3]

        # Calculate distance to center (closer is better for control)
        distance_to_center = abs(r - self.center[0]) + abs(c - self.center[1])
        score = 0
        score -= distance_to_center  # Favor moves closer to the center

        # Favor moves that could push opponent pieces off the board
        adjacent_opponents = 0
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE:
                if game.board[nr][nc] == (PLAYER2 if self.player == PLAYER1 else PLAYER1):
                    adjacent_opponents += 1
                    if self.is_edge_position(nr, nc):
                        score += 3  # Higher score for potential pushes

        # Additional heuristic: Prevent opponent from winning in the next move
        score += adjacent_opponents  # Favor moves near opponent pieces
        return score

    def is_edge_position(self, r, c):
        """Checks if a position is at the edge of the board."""
        return r == 0 or r == BOARD_SIZE - 1 or c == 0 or c == BOARD_SIZE - 1

    def get_best_move(self, game):
        """Chooses the move based on MCTS if possible, otherwise falls back to heuristic."""
        if game.p1_pieces < NUM_PIECES:
            # Use MCTS for early game exploration
            return self.mcts(game)
        else:
            # Use heuristics for late-game positioning
            possible_moves = self.get_possible_moves(game)
            best_move = None
            best_score = float('-inf')
            for move in possible_moves:
                score = self.evaluate_move(game, move)
                if score > best_score:
                    best_score = score
                    best_move = move
            return best_move if best_move is not None else random.choice(possible_moves)