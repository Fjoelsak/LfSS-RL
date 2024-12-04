import numpy as np

from src.TicTacToe.TicTacToe import TicTacToe

def test_isDraw():
    # Example: a 3x3-array (completely filled, no winner)
    obs = np.array([
        [[1, 0], [0, 1], [1, 0]],
        [[1, 0], [1, 0], [0, 1]],
        [[0, 1], [1, 0], [0, 1]],
    ])
    # Expected result: draw
    isDraw = TicTacToe.isDraw(obs)
    isWon, winner = TicTacToe.isWon(obs)
    assert isDraw == True, f"Erwartet: True, erhalten: {isDraw}"
    assert isWon == False, f"Erwartet: False, erhalten: {isWon}"
    assert winner == None


