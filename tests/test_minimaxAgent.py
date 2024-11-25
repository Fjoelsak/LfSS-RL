import numpy as np

from src.HeuristicPlanning.minimaxAgent import minimaxAgent

def test_isDraw():
    agent = minimaxAgent(1)
    # Example: a 3x3-array (completely filled, no winner)
    obs = np.array([
        [[1, 0], [0, 1], [1, 0]],
        [[1, 0], [1, 0], [0, 1]],
        [[0, 1], [1, 0], [0, 1]],
    ])
    # Expected result: draw
    result = agent._is_draw(obs)
    assert result == True, f"Erwartet: True, erhalten: {result}"
    assert agent._evaluate_winner(obs) == None

def test_Eval():
    agent = minimaxAgent(1)

    obs = np.array([
        [[1, 0], [1, 0], [1, 0]],
        [[1, 0], [1, 0], [0, 1]],
        [[0, 0], [1, 0], [1, 0]],
    ])

    result = agent._eval(obs, 0)
    print(result)
    assert result == 3

    obs = np.array([
        [[0, 0], [0, 1], [0, 0]],
        [[0, 0], [1, 0], [0, 0]],
        [[0, 0], [0, 0], [0, 0]],
    ])

    result = agent._eval(obs, 0)
    print(result)
    assert result == 2

    obs = np.array([
        [[0, 0], [1, 0], [0, 0]],
        [[0, 0], [0, 1], [0, 0]],
        [[0, 0], [0, 0], [0, 0]],
    ])

    result = agent._eval(obs, 0)
    print(result)
    assert result == -2

test_Eval()

