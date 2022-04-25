import unittest
import numpy as np
from tetris import get_all_possible_game_states
from pentominos import O_pentomino, I_pentomino, S_pentomino, Z_pentomino, L_pentomino, J_pentomino, T_pentomino


class Testing(unittest.TestCase):
    def test_state_inequality(self):
        board = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        game_state_a = {'board': board, 'score': 0, 'hold': None}
        board2 = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        game_state_b = {'board': board2, 'score': 0, 'hold': None}
        self.assertFalse(get_state_equality(game_state_a, game_state_b))

    def test_pentominos_cant_place(self):
        board = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        game_states = [{'board': board, 'score': 0, 'hold': None}]
        states = get_all_possible_game_states(game_states, O_pentomino, O_pentomino, O_pentomino)
        self.assertTrue(len(states) == 0)

    def test_pentominos_cant_place_2(self):
        board = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        game_states = [{'board': board, 'score': 0, 'hold': None}]
        states = get_all_possible_game_states(game_states, S_pentomino, S_pentomino, S_pentomino)
        self.assertTrue(len(states) == 0)

    def test_board_can_clear_rows_and_get_points(self):
        board = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
        game_states = [{'board': board, 'score': 0, 'hold': None}]
        states = get_all_possible_game_states(game_states, I_pentomino, I_pentomino, I_pentomino)
        expected_state = {'board': np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.int64), 'score': 1, 'hold': np.array([[1], [1], [1], [1]])}
        self.assertTrue(is_state_in_states(expected_state, states))


def get_state_equality(state_a, state_b):
    if state_a['score'] != state_b['score']:
        return False
    if not np.array_equal(state_a['board'], state_b['board']):
        return False
    if not np.array_equal(state_a['hold'], state_b['hold']):
        return False
    return True


def is_state_in_states(state_a, states):
    return any((get_state_equality(state_a, state_b)) for state_b in states)


if __name__ == '__main__':
    unittest.main()
