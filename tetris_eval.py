import random
import statistics
import numpy as np

import os
import sys
import matplotlib.pyplot as plt
from tetris_env import TetrisEnviroment
from tetris import get_all_possible_game_states, is_game_over

# a few naive ranking agents

def naive_state_scoring_agent(state, available_pentominos):
    return state['score']


def naive_state_scoring_agent2(state, available_pentominos):
    return -state['board'].max()


def naive_state_scoring_agent3(state, available_pentominos):
    return -sum(state['board'])


# how to use the fitness function to test an agent
def example_fitness():
    test_env = TetrisEnviroment(random.randint(0, 99**999))
    # more epochs will be slower, but produce a more accurate score
    # consider increasing epochs if variance is too high
    fitness_score = test_env.compute_fitness_score(naive_state_scoring_agent, 10)
    print(f'This agent has a fitness score of {fitness_score}.')

    # how to visualize the performance of an agent

    scores = []
    test_env = TetrisEnviroment(random.randint(0, 99**999))
    for i in range(100):
        score = test_env.run_game(naive_state_scoring_agent2)
        test_env.change_seed(random.randint(0, 99**999))
        scores.append(score)
        print(f'EPOCH {i} COMPLETED - SCORED {score}')

    smallest = min(scores)
    largest = max(scores)

    print(statistics.mean(scores))
    
    plt.bar(*np.unique(scores, return_counts=True))
    plt.ylabel('Occurrences')
    plt.xlabel('Score')
    plt.gca().set_xticks(range(smallest, largest+1, 10))
    plt.show()


if __name__ == '__main__':
    example_fitness()