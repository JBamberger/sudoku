#include "SudokuSolver.h"

#include <algorithm>
#include <assert.h>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <set>
#include <string>
#include <vector>
// template<int GRID_SIZE>

enum class SolveState
{
    SOLVED,
    UNMODIFIED,
    MODIFIED,
    UNSOLVABLE
};

struct Sudoku
{
    //    static_assert(GRID_SIZE == 9, "At the moment only grid_size == 9 is supported");

    static constexpr int BOX_SIZE = 3;
    static constexpr int SIZE = 9;
    static constexpr int NUM_CELLS = 81;

    SudokuGrid grid;
    //    std::array<std::set<int>, NUM_CELLS> candidates;

    static int from2d(int row, int col) { return row * SIZE + col; }

    static std::tuple<int, int> boxCoordinates(int row, int col)
    {
        return std::make_tuple(row / BOX_SIZE, col / BOX_SIZE);
    }

    explicit Sudoku(const SudokuGrid& grid)
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

    SolveState solveTrivial()
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
                    return SolveState::MODIFIED;
                } else if (candidates.empty()) {
                    return SolveState::UNSOLVABLE;
                }
            }
        }

        return solved ? SolveState::SOLVED : SolveState::UNMODIFIED;
    }
};

template<int GRID_SIZE>
struct SudokuSolver::Impl
{
    std::unique_ptr<SudokuGrid> solve(const SudokuGrid& sudokuGrid)
    {
        Sudoku sudoku(sudokuGrid);
        return _solve(sudoku);
    }
    std::unique_ptr<SudokuGrid> _solve(Sudoku& sudoku)
    {
//        std::vector<Sudoku> stack;
//        stack.push_back(sudoku);
//
//        while (!stack.empty()) {
//            // solve trivial
//            // select cell
//            // compute candidates
//            // for each candidate:
//            //  create copy and try candidate
//            //  on success return
//            //  on error backtrack
//        }

        while (true) {
            SolveState solveResult = sudoku.solveTrivial();
            if (solveResult == SolveState::UNMODIFIED) {
                // Not changed, need to try a cell value and backtrack if erroneous
                break;
            } else if (solveResult == SolveState::SOLVED) {
                return std::make_unique<SudokuGrid>(sudoku.grid);
            } else if (solveResult == SolveState::UNSOLVABLE) {
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
            Sudoku sudokuCopy(sudoku);
            sudokuCopy.at(bestRow, bestCol) = candidate;

            auto solution = _solve(sudokuCopy);
            if (solution != nullptr) {
                return solution;
            }
            // Not a solution, backtrack.
        }

        return nullptr;
    }
    bool selectNextCell(Sudoku& sudoku, int& bestRow, int& bestCol) const
    {
        size_t minCandidates = Sudoku::SIZE + 1;
        bestRow = -1;
        bestCol = -1;
        for (int row = 0; row < Sudoku::SIZE; row++) {
            for (int col = 0; col < Sudoku::SIZE; col++) {
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
        return minCandidates <= Sudoku::SIZE;
    }
};

SudokuSolver::SudokuSolver()
  : pimpl{ std::make_unique<Impl<9>>() }
{}
SudokuSolver&
SudokuSolver::operator=(SudokuSolver&&) noexcept = default;
SudokuSolver::SudokuSolver(SudokuSolver&&) noexcept = default;
SudokuSolver::~SudokuSolver() = default;

std::unique_ptr<SudokuGrid>
SudokuSolver::solve(const SudokuGrid& sudoku) const
{
    //    const auto& msg = pimpl->hasConstraintViolations(sudoku);

    //    if (msg == nullptr) {
    //        return nullptr;
    //    }
    //
    //    pimpl->solveTrivialCells(sudoku);
    return pimpl->solve(sudoku);
}
