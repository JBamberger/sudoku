
#include <iostream>
#include <numeric>
#include <vector>

#include "ConstraintSudokuSolver.h"

enum class ConstraintSolverState
{
    SOLVED,
    UNMODIFIED,
    MODIFIED,
    UNSOLVABLE
};

struct ConstraintSolverSudoku
{
    static constexpr int BOX_SIZE = 3;
    static constexpr int SIZE = 9;
    static constexpr int NUM_CELLS = 81;

    SudokuGrid grid;

    static int from2d(int row, int col) { return row * SIZE + col; }

    static std::tuple<int, int> boxCoordinates(int row, int col)
    {
        return std::make_tuple(row / BOX_SIZE, col / BOX_SIZE);
    }

    explicit ConstraintSolverSudoku(const SudokuGrid& grid)
      : grid(grid)
    {}

    int& at(int row, int col) { return grid[from2d(row, col)]; }

    bool isCellFilled(int row, int col) { return at(row, col) > 0; }

    std::vector<int> getCellPossibilities(int row, int col)
    {
        std::array<bool, SIZE + 1> candidates{};
        std::fill(std::begin(candidates), std::end(candidates), true);

        // check row
        for (int i = 0; i < SIZE; i++) {
            candidates[at(i, col)] = false;
        }

        // check col
        for (int i = 0; i < SIZE; i++) {
            candidates[at(row, i)] = false;
        }

        // check box
        auto [boxRow, boxCol] = boxCoordinates(row, col);
        for (int i = boxRow * BOX_SIZE; i < boxRow * BOX_SIZE + BOX_SIZE; i++) {
            for (int j = boxCol * BOX_SIZE; j < boxCol * BOX_SIZE + BOX_SIZE; j++) {
                candidates[at(i, j)] = false;
            }
        }

        std::vector<int> numbers;
        for (int num = 1; num <= SIZE; num++) {
            if (candidates[num]) {
                numbers.push_back(num);
            }
        }

        return numbers;
    }

    ConstraintSolverState solveTrivial()
    {
        bool solved = true;
        for (int row = 0; row < SIZE; row++) {
            for (int col = 0; col < SIZE; col++) {
                if (isCellFilled(row, col)) {
                    // Cell is already filled. No further checks necessary.
                    continue;
                }
                solved = false;
                auto candidates = getCellPossibilities(row, col);

                if (candidates.size() == 1) {
                    grid[from2d(row, col)] = candidates.front();
                    return ConstraintSolverState::MODIFIED;
                } else if (candidates.empty()) {
                    return ConstraintSolverState::UNSOLVABLE;
                }
            }
        }

        return solved ? ConstraintSolverState::SOLVED : ConstraintSolverState::UNMODIFIED;
    }
};

static bool
selectNextCell(ConstraintSolverSudoku& sudoku, int& bestRow, int& bestCol)
{
    size_t minCandidates = ConstraintSolverSudoku::SIZE + 1;
    bestRow = -1;
    bestCol = -1;
    for (int row = 0; row < ConstraintSolverSudoku::SIZE; row++) {
        for (int col = 0; col < ConstraintSolverSudoku::SIZE; col++) {
            if (sudoku.isCellFilled(row, col)) {
                // Cell is already filled. No further checks necessary.
                continue;
            }
            size_t numCandidates = sudoku.getCellPossibilities(row, col).size();
            if (1 < numCandidates && numCandidates < minCandidates) {
                bestRow = row;
                bestCol = col;
                minCandidates = numCandidates;
            }
        }
    }
    return minCandidates <= ConstraintSolverSudoku::SIZE;
}

static std::unique_ptr<SudokuGrid>
_solve(ConstraintSolverSudoku& sudoku)
{
    while (true) {
        ConstraintSolverState solveResult = sudoku.solveTrivial();
        if (solveResult == ConstraintSolverState::UNMODIFIED) {
            // Not changed, need to try a cell value and backtrack if erroneous
            break;
        } else if (solveResult == ConstraintSolverState::SOLVED) {
            return std::make_unique<SudokuGrid>(sudoku.grid);
        } else if (solveResult == ConstraintSolverState::UNSOLVABLE) {
            return nullptr;
        }
    }

    int bestRow;
    int bestCol;
    if (!selectNextCell(sudoku, bestRow, bestCol)) {
        std::cout << "Error: invalid sudoku" << std::endl;
        return nullptr;
    }

    const auto candidates = sudoku.getCellPossibilities(bestRow, bestCol);
    for (const auto candidate : candidates) {
        ConstraintSolverSudoku sudokuCopy(sudoku);
        sudokuCopy.at(bestRow, bestCol) = candidate;

        auto solution = _solve(sudokuCopy);
        if (solution != nullptr) {
            return solution;
        }
        // Not a solution, backtrack.
    }

    return nullptr;
}

SudokuConstraintSolver::~SudokuConstraintSolver() = default;

std::unique_ptr<SudokuGrid>
SudokuConstraintSolver::solve(const SudokuGrid& sudokuGrid) const
{
    ConstraintSolverSudoku sudoku(sudokuGrid);
    return _solve(sudoku);
}
