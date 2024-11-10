# heuristicAgent.py
import numpy as np
import time
from PushBattle import Game, PLAYER1, PLAYER2, EMPTY, BOARD_SIZE, _torus

class ImprovedAgent:
    def __init__(self, depth=3):
        self.player = PLAYER1
        self.depth = depth

    def set_player(self, player):
        self.player = player

    def evaluate_board(self, game):
        """
        Heuristic evaluation function for the board.
        Positive values favor PLAYER1, negative values favor PLAYER2.
        """
        score = 0

        # Piece count
        p1_pieces = np.sum(game.board == PLAYER1)
        p2_pieces = np.sum(game.board == PLAYER2)
        score += (p1_pieces - p2_pieces) * 10

        # Potential winning moves
        score += (self.evaluate_potential_wins(game, PLAYER1) - self.evaluate_potential_wins(game, PLAYER2)) * 100

        # Blocking opponent's winning moves
        score += (self.evaluate_blocking_moves(game, PLAYER1) - self.evaluate_blocking_moves(game, PLAYER2)) * 50

        # Setting up two in a row
        score += (self.evaluate_two_in_a_row(game, PLAYER1) - self.evaluate_two_in_a_row(game, PLAYER2)) * 30

        # Blocking opponent's two in a row
        score += (self.evaluate_blocking_two_in_a_row(game, PLAYER1) - self.evaluate_blocking_two_in_a_row(game, PLAYER2)) * 20

        # Center control
        score += (self.evaluate_center_control(game, PLAYER1) - self.evaluate_center_control(game, PLAYER2)) * 40

        return score

    def evaluate_potential_wins(self, game, player):
        """
        Evaluate potential winning moves for the given player.
        """
        potential_wins = 0
        # Check rows, columns, and diagonals for potential wins
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                if self.check_potential_win(game, player, row, col):
                    potential_wins += 1
        return potential_wins

    def check_potential_win(self, game, player, row, col):
        """
        Check if placing a piece at (row, col) can create a row of three pieces.
        """
        directions = [(-1, 0), (0, -1), (-1, -1), (-1, 1)]
        for dr, dc in directions:
            count = 0
            for i in range(-2, 3):
                r, c = _torus(row + i * dr, col + i * dc)
                if game.board[r][c] == player:
                    count += 1
                else:
                    count = 0
                if count == 3:
                    return True
        return False

    def evaluate_blocking_moves(self, game, player):
        """
        Evaluate moves that can block the opponent from creating a row of three pieces.
        """
        opponent = PLAYER2 if player == PLAYER1 else PLAYER1
        blocking_moves = 0
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                if self.check_potential_win(game, opponent, row, col):
                    blocking_moves += 1
        return blocking_moves

    def evaluate_two_in_a_row(self, game, player):
        """
        Evaluate potential two in a row setups for the given player.
        """
        two_in_a_row = 0
        directions = [(-1, 0), (0, -1), (-1, -1), (-1, 1)]
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                for dr, dc in directions:
                    count = 0
                    for i in range(-1, 2):
                        r, c = _torus(row + i * dr, col + i * dc)
                        if game.board[r][c] == player:
                            count += 1
                        else:
                            count = 0
                        if count == 2:
                            two_in_a_row += 1
        return two_in_a_row

    def evaluate_blocking_two_in_a_row(self, game, player):
        """
        Evaluate moves that can block the opponent from setting up two in a row.
        """
        opponent = PLAYER2 if player == PLAYER1 else PLAYER1
        blocking_two_in_a_row = 0
        directions = [(-1, 0), (0, -1), (-1, -1), (-1, 1)]
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                for dr, dc in directions:
                    count = 0
                    for i in range(-1, 2):
                        r, c = _torus(row + i * dr, col + i * dc)
                        if game.board[r][c] == opponent:
                            count += 1
                        else:
                            count = 0
                        if count == 2:
                            blocking_two_in_a_row += 1
        return blocking_two_in_a_row

    def evaluate_center_control(self, game, player):
        """
        Evaluate control of the center of the board.
        """
        center_control = 0
        center_positions = [(3, 3), (3, 4), (4, 3), (4, 4)]
        for r, c in center_positions:
            if game.board[r][c] == player:
                center_control += 1
        return center_control

    def minimax(self, game, depth, alpha, beta, maximizing_player, start_time, time_limit):
        """
        Minimax algorithm with alpha-beta pruning and time limit.
        """
        if depth == 0 or game.check_winner() != EMPTY or time.time() - start_time > time_limit:
            return self.evaluate_board(game)

        if maximizing_player:
            max_eval = float('-inf')
            for move in self.generate_possible_moves(game):
                if time.time() - start_time > time_limit:
                    break
                new_game = self.simulate_move(game, move)
                eval = self.minimax(new_game, depth - 1, alpha, beta, False, start_time, time_limit)
                max_eval = max(max_eval, eval)
                alpha = max(alpha, eval)
                if beta <= alpha or time.time() - start_time > time_limit:
                    break
            return max_eval
        else:
            min_eval = float('inf')
            for move in self.generate_possible_moves(game):
                if time.time() - start_time > time_limit:
                    break
                new_game = self.simulate_move(game, move)
                eval = self.minimax(new_game, depth - 1, alpha, beta, True, start_time, time_limit)
                min_eval = min(min_eval, eval)
                beta = min(beta, eval)
                if beta <= alpha or time.time() - start_time > time_limit:
                    break
            return min_eval

    def get_best_move(self, game, time_limit=4):
        """
        Determine the best move based on the minimax algorithm with alpha-beta pruning and time limit.
        """
        best_score = float('-inf')
        best_move = None
        start_time = time.time()

        for move in self.generate_possible_moves(game):
            if time.time() - start_time > time_limit:
                break
            new_game = self.simulate_move(game, move)
            score = self.minimax(new_game, self.depth - 1, float('-inf'), float('inf'), False, start_time, time_limit)
            if score > best_score:
                best_score = score
                best_move = move

        return best_move

    def generate_possible_moves(self, game):
        """
        Generate all possible moves for the current player.
        """
        moves = []

        # If the player has less than 8 pieces, generate placement moves
        if (self.player == PLAYER1 and game.p1_pieces < 8) or (self.player == PLAYER2 and game.p2_pieces < 8):
            for r in range(BOARD_SIZE):
                for c in range(BOARD_SIZE):
                    if game.is_valid_placement(r, c):
                        moves.append([r, c])
        else:
            # Generate movement moves
            for r0 in range(BOARD_SIZE):
                for c0 in range(BOARD_SIZE):
                    if game.board[r0][c0] == self.player:
                        for r1 in range(BOARD_SIZE):
                            for c1 in range(BOARD_SIZE):
                                if game.is_valid_move(r0, c0, r1, c1):
                                    moves.append([r0, c0, r1, c1])

        # Prioritize moves towards the center
        center_positions = [(3, 3), (3, 4), (4, 3), (4, 4)]
        moves.sort(key=lambda move: self.move_priority(move, center_positions), reverse=True)

        return moves

    def move_priority(self, move, center_positions):
        """
        Calculate the priority of a move based on its proximity to the center.
        """
        if len(move) == 2:
            r, c = move
        elif len(move) == 4:
            r, c = move[2], move[3]
        else:
            return 0

        return -min(abs(r - cr) + abs(c - cc) for cr, cc in center_positions)

    def simulate_move(self, game, move):
        """
        Simulate the given move and return the new game state.
        """
        new_game = Game.from_dict(game.to_dict())

        if len(move) == 2:
            new_game.place_checker(move[0], move[1])
        elif len(move) == 4:
            new_game.move_checker(move[0], move[1], move[2], move[3])

        return new_game