import numpy as np

from src.HeuristicPlanning.minimaxAgent import minimaxAgent

def test_Eval():
    agent = minimaxAgent(None, 0)

    obs = np.array([
        [[1, 0], [1, 0], [1, 0]],
        [[1, 0], [1, 0], [0, 1]],
        [[0, 0], [1, 0], [1, 0]],
    ])

    result = agent._eval(obs)
    print(result)
    assert result == 9

    obs = np.array([
        [[0, 0], [0, 1], [0, 0]],
        [[0, 0], [1, 0], [0, 0]],
        [[0, 0], [0, 0], [0, 0]],
    ])

    result = agent._eval(obs)
    print(result)
    assert result == 2

    obs = np.array([
        [[0, 0], [1, 0], [0, 0]],
        [[0, 0], [0, 1], [0, 0]],
        [[0, 0], [0, 0], [0, 0]],
    ])

    result = agent._eval(obs)
    print(result)
    assert result == -2
