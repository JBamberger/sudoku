from typing import List
from unittest import TestCase
import timeit

from solver.sudoku_solver import solve_sudoku


def read_test_sudokus():
    def to_sudoku(line: str) -> List[List[int]]:
        nums = list(map(lambda x: int(x) if x != '.' else 0, line.strip()))
        board = [nums[i:i + 9] for i in range(0, 81, 9)]
        return board

    sudokus = []
    with open('test_sudokus.txt', 'r', encoding='utf8') as f:
        for line in f:
            sudokus.append(line.split(';'))
    sudokus = [(to_sudoku(board), to_sudoku(solution)) for board, solution in sudokus]
    return sudokus


class Test(TestCase):

    def test_solve_sudoku(self):
        sudokus = read_test_sudokus()

        for sudoku, solution in sudokus:
            proposed_solution = solve_sudoku(sudoku)

            self.assertEqual(solution, proposed_solution)

    def xtest_timing(self):
        sudokus = read_test_sudokus()

        for sudoku, solution in sudokus:
            print(timeit.timeit(lambda: solve_sudoku(sudoku.copy()), number=10))
            # proposed_solution = solve_sudoku(sudoku)
