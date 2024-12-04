from TicTacToe import TicTacToe
from src.HeuristicPlanning.alphaBetaAgent import alphaBetaAgent
from src.HeuristicPlanning.minimaxAgent import minimaxAgent
import time


game = TicTacToe()
obs = game.reset()
ai_agent = alphaBetaAgent(game, player = 1)

done = False
while not done:
    for agent in game.agents:
        if game.whoseTurn == 0:
            action = game.get_user_action()
            #action = ai_agent.find_best_move(obs)
        else:
            #action = game.get_user_action()
            action = ai_agent.find_best_move(obs, 3)

        obs, done, info = game.step(action)

        if done:
            if info['isWon']:
                print("Player", info['Winner'], "wins!")
            elif info['isDraw']:
                print("This is a draw!")
            break

time.sleep(1)