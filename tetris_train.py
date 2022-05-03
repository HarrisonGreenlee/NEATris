from tetris_env import TetrisEnviroment
import neat
import random
import numpy as np


def run_tetris(genomes, config):
    game = TetrisEnviroment(random.randint(0, 99**999))
    for _, g in genomes:
        net = neat.nn.FeedForwardNetwork.create(g, config)

        def state_evaluator(board_state, available_pentominos):
            inputs = [column_height/20 for column_height in board_state['board']] + [board_state['board'].max()/20] + available_pentominos + [board_state['score']/9999] + [sum(board_state['board'])/200]
            return net.activate(inputs)

        g.fitness = game.compute_fitness_score(state_evaluator, 20)


if __name__ == '__main__':
    config_path = "./config-feedforward.txt"
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)
    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    # save a checkpoint every 50 generations
    p.add_reporter(neat.Checkpointer(100))
    p.run(run_tetris, 1000)
