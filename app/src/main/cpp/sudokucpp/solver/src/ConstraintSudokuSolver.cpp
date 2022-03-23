
#include <array>
#include <iostream>
#include <numeric>
#include <utility>
#include <vector>

#include "ConstraintSudokuSolver.h"

enum class ConstraintSolverState
{
    SOLVED,
    UNMODIFIED,
    MODIFIED,
    UNSOLVABLE
};

std::vector<int>
getCellPossibilities(const SudokuGrid& grid, int row, int col)
{
    std::vector<bool> candidates(grid.sideLen + 1);
    std::fill(std::begin(candidates), std::end(candidates), true);

    // check row
    for (int i = 0; i < grid.sideLen; i++) {
        candidates.at(grid.at(i, col)) = false;
    }

    // check col
    for (int i = 0; i < grid.sideLen; i++) {
        candidates.at(grid.at(row, i)) = false;
    }

    // check box
    size_t boxRow = row / grid.boxLen;
    size_t boxCol = col / grid.boxLen;
    for (size_t i = boxRow * grid.boxLen; i < boxRow * grid.boxLen + grid.boxLen; i++) {
        for (size_t j = boxCol * grid.boxLen; j < boxCol * grid.boxLen + grid.boxLen; j++) {
            candidates.at(grid.at(i, j)) = false;
        }
    }

    std::vector<int> numbers;
    for (int num = 1; num <= grid.sideLen; num++) {
        if (candidates.at(num)) {
            numbers.push_back(num);
        }
    }

    return numbers;
}

ConstraintSolverState
solveTrivialCells(SudokuGrid& grid)
{
    bool solved = true;
    for (int row = 0; row < grid.sideLen; row++) {
        for (int col = 0; col < grid.sideLen; col++) {
            if (grid.isCellFilled(row, col)) {
                // Cell is already filled. No further checks necessary.
                continue;
            }
            solved = false;
            auto candidates = getCellPossibilities(grid, row, col);

            if (candidates.size() == 1) {
                grid.at(row, col) = candidates.front();
                return ConstraintSolverState::MODIFIED;
            } else if (candidates.empty()) {
                return ConstraintSolverState::UNSOLVABLE;
            }
        }
    }

    return solved ? ConstraintSolverState::SOLVED : ConstraintSolverState::UNMODIFIED;
}

ConstraintSolverState
solveTrivialCellsIter(SudokuGrid& grid)
{
    ConstraintSolverState lastState;
    while ((lastState = solveTrivialCells(grid)) == ConstraintSolverState::MODIFIED) {
    }

    return lastState;
}

bool
selectNextCell(const SudokuGrid& grid, int& bestRow, int& bestCol)
{
    size_t minCandidates = grid.sideLen + 1;
    bestRow = -1;
    bestCol = -1;
    for (int row = 0; row < grid.sideLen; row++) {
        for (int col = 0; col < grid.sideLen; col++) {
            if (grid.isCellFilled(row, col)) {
                // Cell is already filled. No further checks necessary.
                continue;
            }
            size_t numCandidates = getCellPossibilities(grid, row, col).size();
            if (1 < numCandidates && numCandidates < minCandidates) {
                bestRow = row;
                bestCol = col;
                minCandidates = numCandidates;
            }
        }
    }
    return minCandidates <= grid.sideLen;
}

std::unique_ptr<SudokuGrid>
_solve(SudokuGrid& grid)
{
    switch (solveTrivialCellsIter(grid)) {
        case ConstraintSolverState::MODIFIED:
            assert(false); // This cannot happen because it would result in another iteration of solveTrivialCells()
            break;
        case ConstraintSolverState::UNMODIFIED:
            // Not changed, need to try a cell value and backtrack if erroneous
            break;
        case ConstraintSolverState::SOLVED:
            return std::make_unique<SudokuGrid>(grid);
        case ConstraintSolverState::UNSOLVABLE:
            return nullptr;
    }

    int bestRow;
    int bestCol;
    if (!selectNextCell(grid, bestRow, bestCol)) {
        std::cout << "Error: invalid sudoku" << std::endl;
        return nullptr;
    }

    const auto candidates = getCellPossibilities(grid, bestRow, bestCol);
    for (const auto candidate : candidates) {
        SudokuGrid sudokuCopy(grid);
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
    SudokuGrid sudoku(sudokuGrid);
    return _solve(sudoku);
}
