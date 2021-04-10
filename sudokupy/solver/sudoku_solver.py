from copy import deepcopy
from typing import List, Set, Optional

from solver.sudoku import Sudoku


def _solve(game, depth=0) -> Optional[List[List[int]]]:
    while True:
        edited = False

        solved = True
        for i in range(game.size):
            for j in range(game.size):
                if game.grid[i][j] != 0:
                    continue

                solved = False
                candidates: Set[int] = game.candidates[i][j]

                # The board cannot be solved because no candidates are left for cell. Abort.
                if len(candidates) == 0:
                    return None

                # There is a unique solution for this cell. Set and propagate information.
                if len(candidates) == 1:
                    game.set_position(i, j, next(iter(candidates)))
                    edited = True

        # Try again after edits
        if edited:
            continue

        if solved:
            return game.grid.copy()

        # Could not find a simple edit and board is not solved yed. Do backtracking search.
        # Find a cell with few remaining options to perform most efficient backtracking.
        min_guesses = (game.size + 1, -1)
        for i in range(game.size):
            for j in range(game.size):
                guesses = len(game.candidates[i][j])
                # not already solved and better than previous best
                if 1 < guesses < min_guesses[0]:
                    min_guesses = (guesses, (i, j))

        # Try all candidates, one by one until a solution is found.
        i, j = min_guesses[1]
        for y in game.candidates[i][j]:
            # recurse with copy, backtrack if solving failed
            backtrack_copy = deepcopy(game)
            backtrack_copy.set_position(i, j, y)

            solution = _solve(backtrack_copy, depth=depth + 1)
            if solution is not None:
                # Found a solution, the assumption was correct.
                return solution

        return None


def solve_sudoku(grid: List[List[int]]) -> Optional[List[List[int]]]:
    sudoku = Sudoku(grid)

    possible, message = sudoku.has_constraint_violations()
    if not possible:
        print('Sudoku constraint violation: %s' % message)
        return None

    sudoku.solve_trivial_cells()
    solution = _solve(sudoku, depth=0)

    return solution
