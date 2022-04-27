import random
import statistics
import numpy as np
from tetris import get_all_possible_game_states, is_game_over
from pentominos import O_pentomino, I_pentomino, S_pentomino, Z_pentomino, L_pentomino, J_pentomino, T_pentomino
from more_itertools import sort_together, locate  # pip install more-itertools


class TetrisEnviroment:
    def __init__(self, seed):
        # todo - seed should make pentomino recommendations consistent across different tests
        # right now each agent is getting different pentominos which makes it less accurate to compare their performance
        self.seed = seed
        # each pentomino in the bag must be placed once before randomization can repeat
        self.pentominos_bag = [O_pentomino, I_pentomino, S_pentomino, Z_pentomino, L_pentomino, J_pentomino, T_pentomino]
        self.available_pentominos_in_current_bag = [1 for tile in self.pentominos_bag]
        board = np.zeros(shape=10, dtype=int)
        self.game_state = {'board': board, 'score': 0, 'hold': None}

    def get_next_pentomino(self):
        if 1 not in self.available_pentominos_in_current_bag:
            self.available_pentominos_in_current_bag = [1 for _ in self.available_pentominos_in_current_bag]
        available_pentomino_indices = list(locate(self.available_pentominos_in_current_bag, lambda x: x == 1))
        selected_pentomino_index = random.choice(available_pentomino_indices)
        self.available_pentominos_in_current_bag[selected_pentomino_index] = 0
        return self.pentominos_bag[selected_pentomino_index]

    def run_game(self, state_scoring_agent):
        current_pentomino = self.get_next_pentomino()
        next_pentomino = self.get_next_pentomino()
        while not is_game_over(self.game_state):
            states = get_all_possible_game_states(self.game_state, current_pentomino, next_pentomino, None)
            if not states:
                break
            state_scores = [state_scoring_agent(state, self.available_pentominos_in_current_bag) for state in states]
            # reverse the sort because higher scores are better
            ranked_states = sort_together([state_scores, states], reverse=True)[1]
            self.game_state = ranked_states[0]['parent']
            current_pentomino = next_pentomino
            next_pentomino = self.get_next_pentomino()

        return self.game_state['score']

    def compute_fitness_score(self, state_scoring_agent, epochs):
        return statistics.mean(float(self.run_game(state_scoring_agent)) for _ in range(epochs))