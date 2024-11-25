import time
import numpy as np

class minimaxAgent:
    def __init__(self,player):
        self.player = player

    def _is_draw(self, obs):
        """
        Checks if the game is a draw.
        :param obs: The current game state.
        :return: True if it's a draw, otherwise False.
        """
        winner = self._evaluate_winner(obs)
        return not (obs[:, :, 0] + obs[:,:,1] == 0).any() and not((winner == 0) or (winner == 1))

    def _evaluate_winner(self, obs) -> int:
        """
        Checks if there is a winner and returns the winner.
        :param obs: The current game state (3x3x2 array).
        :return: 0 (Player 0 wins), 1 (Player 1 wins), or None (no winner).
        """
        for player in range(2):
            # checking rows
            for row in range(3):
                if all(obs[row, col, player] == 1 for col in range(3)):
                    return player
            # checking columns
            for col in range(3):
                if all(obs[row, col, player] == 1 for row in range(3)):
                    return player
            # checking diagonals
            if all(obs[i, i, player] == 1 for i in range(3)) or \
                    all(obs[i, 2 - i, player] == 1 for i in range(3)):
                return player
        # no winner
        return None

    def _eval(self,obs, player):
        X1, X2, O1, O2 = 0, 0, 0, 0

        # Check rows and columns
        for axis in range(2):  # axis 0 for rows, 1 for columns
            for i in range(3):
                # Get the line (row or column)
                line = np.take(obs[:, :, player], i, axis=axis)
                opponent_line = np.take(obs[:, :, 1 - player], i, axis=axis)

                # Check if the line is not blocked by opponent (no opponent piece in the line)
                if not np.any(opponent_line):
                    # Count the pieces for player X (player) and opponent O (1 - player)
                    if np.sum(line) == 1:
                        X1 += 1
                    elif np.sum(line) == 2:
                        X2 += 1
                # Check for O's in the line
                if not np.any(line):
                    if np.sum(opponent_line) == 1:
                        O1 += 1
                    elif np.sum(opponent_line) == 2:
                        O2 += 1

        # Check diagonals
        diagonals = [
            np.array([obs[i, i, player] for i in range(3)]),  # Main diagonal
            np.array([obs[i, 2 - i, player] for i in range(3)])  # Anti-diagonal
        ]

        # For each diagonal, check if it's blocked by the opponent and count the pieces
        for diag in diagonals:
            opponent_diag = [obs[i, i, 1 - player] for i in range(3)]  # Main diagonal opponent pieces
            if not np.any(opponent_diag) and np.sum(diag) == 1:
                X1 += 1
            elif not np.any(opponent_diag) and np.sum(diag) == 2:
                X2 += 1

            opponent_diag = [obs[i, 2 - i, 1 - player] for i in range(3)]  # Anti-diagonal opponent pieces
            if not np.any(diag) and np.sum(opponent_diag) == 1:
                O1 += 1
            elif not np.any(diag) and np.sum(opponent_diag) == 2:
                O2 += 1

        return 3*X2+X1 - (3*O2+O1)

    def minimax(self, obs, is_maximizing, current_player, depth, max_depth):
        """
        Recursive Minimax-Function.
        :param obs: the current board position (3x3x2-Array).
        :param is_maximizing: True, if the maximizing player is to move
        :param current_player: the player whose turn it is (0 oder 1).
        :return: best value for the current board position
        """
        # checking game situation
        winner = self._evaluate_winner(obs)
        if winner is not None:
            return 1 if winner == self.player else -1
        if self._is_draw(obs):
            return 0

        # Check depth limit
        if depth >= max_depth:
            return self._eval(obs, self.player)

        # maximizing or minimizing logic
        best_value = float('-inf') if is_maximizing else float('inf')
        for row in range(3):
            for col in range(3):
                if obs[row, col, 0] == 0 and obs[row, col, 1] == 0:  # Freies Feld
                    # simulate the move
                    obs[row, col, current_player] = 1
                    value = self.minimax(obs, not is_maximizing, 1 - current_player, depth + 1, max_depth)
                    obs[row, col, current_player] = 0  # undo move

                    # Update best_value
                    if is_maximizing:
                        best_value = max(best_value, value)
                    else:
                        best_value = min(best_value, value)
        return best_value

    def find_best_move(self, obs, max_depth=10):
        """
        Find the best move for the agent.
        :param obs: The current board position (3x3x2-Array).
        :return: index of the best move (0-8).
        """

        starttime = time.time()
        best_value = float('-inf')
        best_move = None

        if np.all(np.array(obs[:,:,:]) == 0):
            best_move = 4
        else:
            for row in range(3):
                for col in range(3):
                    if obs[row, col, 0] == 0 and obs[row, col, 1] == 0:  # empty cell
                        # simulate a move
                        obs[row, col, self.player] = 1
                        move_value = self.minimax(obs,  False, 1 - self.player, 1, max_depth)
                        # print("row, col, value", row, col, move_value)
                        obs[row, col, self.player] = 0  # Rückgängig machen

                        # Update best_move
                        if move_value > best_value:
                            best_value = move_value
                            best_move = row + col * 3

        endtime = time.time()
        print("Minimax agent took", round(endtime - starttime,4), "seconds to search the best move.")
        return best_move