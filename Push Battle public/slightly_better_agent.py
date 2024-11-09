import random
from PushBattle import Game, PLAYER1, PLAYER2, EMPTY, BOARD_SIZE, NUM_PIECES

class ImprovedAgent:
    def __init__(self, player=PLAYER1):
        self.player = player
        self.center = (BOARD_SIZE // 2, BOARD_SIZE // 2)

    def get_possible_moves(self, game):
        """Returns list of all possible moves in current state."""
        moves = []
        current_pieces = game.p1_pieces if game.current_player == PLAYER1 else game.p2_pieces

        if current_pieces < NUM_PIECES:
            # Placement moves
            for r in range(BOARD_SIZE):
                for c in range(BOARD_SIZE):
                    if game.board[r][c] == EMPTY:
                        moves.append((r, c))
        else:
            # Movement moves
            for r0 in range(BOARD_SIZE):
                for c0 in range(BOARD_SIZE):
                    if game.board[r0][c0] == game.current_player:
                        for r1 in range(BOARD_SIZE):
                            for c1 in range(BOARD_SIZE):
                                if game.board[r1][c1] == EMPTY:
                                    moves.append((r0, c0, r1, c1))
        return moves

    def evaluate_move(self, game, move):
        """Assigns a score to each move based on simple heuristics."""
        if len(move) == 2:  # Placement move
            r, c = move
        else:  # Movement move
            r, c = move[2], move[3]

        # Calculate distance to center (lower is better for control)
        distance_to_center = abs(r - self.center[0]) + abs(c - self.center[1])

        # Heuristic score calculation
        score = 0
        score -= distance_to_center  # Favor moves closer to the center

        # Check adjacency to opponent's pieces for possible control/aggression
        adjacent_opponents = 0
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE:
                if game.board[nr][nc] == (PLAYER2 if self.player == PLAYER1 else PLAYER1):
                    adjacent_opponents += 1
        score += adjacent_opponents  # Favor moves near opponent pieces

        return score

    def get_best_move(self, game):
        """Returns the move with the best score based on evaluation."""
        possible_moves = self.get_possible_moves(game)
        
        # Evaluate each move and choose the one with the highest score
        best_move = None
        best_score = float('-inf')
        for move in possible_moves:
            score = self.evaluate_move(game, move)
            if score > best_score:
                best_score = score
                best_move = move

        return best_move if best_move is not None else random.choice(possible_moves)