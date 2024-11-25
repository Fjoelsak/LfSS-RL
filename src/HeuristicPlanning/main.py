from TicTacToe import TicTacToe
from minimaxAgent import minimaxAgent
from alphaBetaAgent import alphaBetaAgent
import time

ai_agent = alphaBetaAgent(player = 1)

game = TicTacToe()
obs = game.reset()

done = False
while not done:
    for agent in game.agents:
        if game.whoseTurn == 0:
            action = game.get_user_action()
            #action = ai_agent.find_best_move(obs)
        else:
            #action = game.get_user_action()
            action = ai_agent.find_best_move(obs)

        obs, done, info = game.step(action)

        if done:
            if info['isWon']:
                print(info['Winner'], "win")
            elif info['isDraw']:
                print("This is a draw!")
            break

time.sleep(1)