import pickle
import neat
import tetris_env
import random

# this could be cleaned up a bit, but we need a visualization of our NN
# so it will work for now
with open('winner.pickle', 'rb') as handle:
    winner = pickle.load(handle)

# load configuration
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     'config-feedforward.txt')

# initialize winner net
net = neat.nn.FeedForwardNetwork.create(winner, config)


game = tetris_env.TetrisEnviroment(random.randint(0, 99 ** 999))


def state_evaluator(board_state, available_pentominos):
    inputs = [column_height / 20 for column_height in board_state['board']] + [
        board_state['board'].max() / 20] + available_pentominos + [board_state['score'] / 9999] + [
                 sum(board_state['board']) / 200]
    return net.activate(inputs)


logger = []
game.run_game(state_evaluator, log=logger)


print(f'Logging a game where {logger[-1]["score"]} lines were cleared.')
with open('log.pickle', 'wb') as handle:
    pickle.dump(logger, handle, protocol=pickle.HIGHEST_PROTOCOL)