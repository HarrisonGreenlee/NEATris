from tetris_env import TetrisEnviroment
import neat
import random
import numpy as np

def run_tetris(genomes, config):
    
    for id, g in genomes:
        game = TetrisEnviroment(random.random())
        
        net = neat.nn.FeedForwardNetwork.create(g, config)

        def state_evaluator(board_state, avalible_pentominos):
            return net.activate(list(board_state['board']) + avalible_pentominos + [board_state['score']])
        
        g.fitness = game.compute_fitness_score(state_evaluator, 10)

if __name__ == '__main__':
    tetris_env = TetrisEnviroment(random.random())
    config_path = "./config-feedforward.txt"
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)
    config
    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.run(run_tetris, 100)