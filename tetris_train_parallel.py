import multiprocessing
import os
import neat
from tetris_env import TetrisEnviroment
import numpy as np
import random
# import visualize TODO

def eval_genome(genome, config):
    game = TetrisEnviroment(random.randint(0, 99 ** 999))
    net = neat.nn.FeedForwardNetwork.create(genome, config)

    def state_evaluator(board_state, available_pentominos):
        inputs = [column_height / 20 for column_height in board_state['board']] + [
            board_state['board'].max() / 20] + available_pentominos + [board_state['score'] / 9999] + [
                     sum(board_state['board']) / 200]
        return net.activate(inputs)

    return game.compute_fitness_score(state_evaluator, 20)

def run(config_file):
    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(50))

    pe = neat.ParallelEvaluator(multiprocessing.cpu_count(), eval_genome)
    winner = p.run(pe.evaluate, 5000)

    print('\nBest genome:\n{!s}'.format(winner))

    # TODO - figure out where visualize is
    #print('\nOutput:')
    #winner_net = neat.nn.FeedForwardNetwork.create(winner, config)


    # visualize.draw_net(config, winner, True)
    # visualize.draw_net(config, winner, True, prune_unused=True)
    # visualize.plot_stats(stats, ylog=False, view=True)
    # visualize.plot_species(stats, view=True)

if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward.txt')
    run(config_path)