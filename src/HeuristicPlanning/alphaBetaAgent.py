from minimaxAgent import minimaxAgent
import numpy as np
import time

class alphaBetaAgent(minimaxAgent):

    def __init__(self, player):
        super().__init__(player)

    def alphabeta(self, obs, is_maximizing, current_player, alpha, beta, depth, max_depth):
        """
        Recursive Alpha-Beta-Pruning function.
        :param obs: the current board position (3x3x2-Array).
        :param is_maximizing: True, if the maximizing player is to move
        :param current_player: the player whose turn it is (0 oder 1).
        :param alpha: the best already guaranteed value for the maximizing player.
        :param beta: the best already guaranteed value for the minimizing player.
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

        if is_maximizing:
            best_value = float('-inf')
            for row in range(3):
                for col in range(3):
                    if obs[row, col, 0] == 0 and obs[row, col, 1] == 0:  # empty cell
                        # simulate the move
                        obs[row, col, current_player] = 1
                        value = self.alphabeta(obs, False, 1 - current_player, alpha, beta, depth + 1, max_depth)
                        obs[row, col, current_player] = 0  # undo move

                        best_value = max(best_value, value)
                        alpha = max(alpha, best_value)

                        # Prune branch
                        if alpha >= beta:
                            break
        else:
            best_value = float('inf')
            for row in range(3):
                for col in range(3):
                    if obs[row, col, 0] == 0 and obs[row, col, 1] == 0:  # empty cell
                        # simulate the move
                        obs[row, col, current_player] = 1
                        value = self.alphabeta(obs, True, 1 - current_player, alpha, beta, depth + 1, max_depth)
                        obs[row, col, current_player] = 0  # undo move

                        best_value = min(best_value, value)
                        beta = min(beta, best_value)

                        # Prune branch
                        if alpha >= beta:
                            break

        return best_value

    def find_best_move(self, obs, max_depth=10):
        """
        Find the best move for the agent using Alpha-Beta Pruning.
        :param obs: The current board position (3x3x2-Array).
        :return: index of the best move (0-8).
        """
        starttime = time.time()
        best_value = float('-inf')
        best_move = None

        if np.all(np.array(obs[:, :, :]) == 0):
            best_move = 4  # Play the center on an empty board
        else:
            alpha = float('-inf')
            beta = float('inf')
            for row in range(3):
                for col in range(3):
                    if obs[row, col, 0] == 0 and obs[row, col, 1] == 0:  # empty cell
                        # simulate a move
                        obs[row, col, self.player] = 1
                        move_value = self.alphabeta(obs, False, 1 - self.player, alpha, beta, 1, max_depth)
                        obs[row, col, self.player] = 0  # undo move

                        # Update best_move
                        if move_value > best_value:
                            best_value = move_value
                            best_move = row + col * 3
                            alpha = max(alpha, best_value)

        endtime = time.time()
        print("Alpha-Beta agent took", round(endtime - starttime, 4), "seconds to search the best move.")
        return best_move
