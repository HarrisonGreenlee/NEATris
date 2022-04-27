import numpy as np
from pentominos import O_pentomino, I_pentomino, S_pentomino, Z_pentomino, L_pentomino, J_pentomino, T_pentomino


def get_unique_rotations(pentomino):
    unique_rotations = []
    for A in [pentomino, np.rot90(pentomino, k=1), np.rot90(pentomino, k=2), np.rot90(pentomino, k=3)]:
        # bad time complexity but there are only 4 elements in the list so it doesn't matter
        if not any(np.array_equal(A, B) for B in unique_rotations):
            unique_rotations.append(A)
    return unique_rotations


def get_docking_map(pentomino):
    docking_map = (pentomino == 1).argmin(axis=0)
    docking_map[docking_map == 0] = len(pentomino)
    return docking_map


def get_surface_map(pentomino):
    return -(pentomino != 0).argmax(axis=0)


def offset_rows(arr):
    n, m = arr.shape
    out = np.zeros((n, n + m), dtype=arr.dtype)
    out[:n, :m] = arr
    return out.reshape(-1)[:-n].reshape(n, n + m - 1)


def get_possible_game_states(game_state, pentomino):
    board = game_state['board']
    sliding_docking_window = np.lib.stride_tricks.sliding_window_view(board, pentomino.shape[1]) + get_docking_map(pentomino)
    all_offsets = offset_rows(sliding_docking_window + get_surface_map(pentomino))
    all_boards = (board * ~all_offsets.astype(bool) + all_offsets)
    possible_boards = all_boards[np.min(sliding_docking_window, axis=1) == np.max(sliding_docking_window, axis=1)]
    cleared_rows = possible_boards.min(axis=1)
    possible_boards = possible_boards - cleared_rows[:,None]
    # create a new partial game state for each possible board
    return [{'board': possible_board, 'score': game_state['score'] + cleared_row} for possible_board, cleared_row in zip(possible_boards, cleared_rows)]


def get_possible_game_states_with_rotation(partial_game_states, pentomino):
    if isinstance(partial_game_states, dict):
        partial_game_states = [partial_game_states]
    if len(partial_game_states) == 0:
        return partial_game_states
    possible_partial_game_states = []
    for partial_game_state in partial_game_states:
        for rotated_pentomino in get_unique_rotations(pentomino):
            possible_partial_game_states.extend(get_possible_game_states(partial_game_state, rotated_pentomino))
    return possible_partial_game_states  # still missing the pentomino that is being held


def assemble_partial_states(partial_states, pentomino_hold, parent_state=None):
    for partial_state in partial_states:
        partial_state['hold'] = pentomino_hold
        partial_state['parent'] = parent_state
    return partial_states


def get_all_possible_game_states(full_game_states, pentomino_1, pentomino_2, pentomino_hold):
    all_possible_states = []
    # try placing both pentominos without holding
    parent_states = assemble_partial_states(get_possible_game_states_with_rotation(full_game_states, pentomino_1), pentomino_hold)
    for parent_state in parent_states:
        child_states = assemble_partial_states(get_possible_game_states_with_rotation(parent_state, pentomino_2), pentomino_hold, parent_state)
        all_possible_states.extend(child_states)
    if pentomino_hold is not None:
        # try placing the hold and then placing the first pentomino
        parent_states = assemble_partial_states(get_possible_game_states_with_rotation(full_game_states, pentomino_hold), pentomino_1)
        for parent_state in parent_states:
            child_states = assemble_partial_states(get_possible_game_states_with_rotation(parent_state, pentomino_1), pentomino_2, parent_state)
            all_possible_states.extend(child_states)
        # try placing the hold and then placing the second pentomino
        parent_states = assemble_partial_states(get_possible_game_states_with_rotation(full_game_states, pentomino_hold), pentomino_1)
        for parent_state in parent_states:
            child_states = assemble_partial_states(get_possible_game_states_with_rotation(parent_state, pentomino_2), pentomino_1, parent_state)
            all_possible_states.extend(child_states)
    else:
        # use empty hold to swap pentominos, then place both
        # this is a rare case, but valuable for not getting stuck at the start of the game
        parent_states = assemble_partial_states(get_possible_game_states_with_rotation(full_game_states, pentomino_2), pentomino_1)
        for parent_state in parent_states:
            # we have no way of knowing what our hold will be in this case
            # so we can only set it to none
            # this is only for NN evaluation purposes
            # as long as we only advance by one node at a time our sim will still follow the rules of tetris
            child_states = assemble_partial_states(get_possible_game_states_with_rotation(parent_state, pentomino_1), None, parent_state)
            all_possible_states.extend(child_states)
    return all_possible_states


def is_game_over(game_state):
    return game_state['board'].max() > 20


if __name__ == '__main__':
    # board_state structure: board (1D array of numbers), hold (2D array containing a pentomino or None), points (number), parent state (the state that produced this state)
    # in accordance with standard tetris rules we start with just a single board state - the empty board
    board = np.zeros(shape=10, dtype=int)
    game_states = [{'board':board, 'score':0, 'hold':None}]
    states = get_all_possible_game_states(game_states, O_pentomino, O_pentomino, L_pentomino)
    print(states)

