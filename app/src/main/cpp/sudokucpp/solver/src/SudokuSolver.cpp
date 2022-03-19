//
// Created by jannik on 08.08.2021.
//

#include "SudokuSolver.h"
#include "dlx.h"

#include <algorithm>
#include <cassert>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <vector>

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

class SudokuConstraintSolver : public SudokuSolver
{
    std::unique_ptr<SudokuGrid> _solve(ConstraintSolverSudoku& sudoku) const
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

    static bool selectNextCell(ConstraintSolverSudoku& sudoku, int& bestRow, int& bestCol)
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

  public:
    ~SudokuConstraintSolver() override = default;

    [[nodiscard]] std::unique_ptr<SudokuGrid> solve(const SudokuGrid& sudokuGrid) const override
    {
        ConstraintSolverSudoku sudoku(sudokuGrid);
        return _solve(sudoku);
    }
};

static size_t sudokuSide = 9;
static size_t boxSide = 3;

/**
 * Computes the exact cover matrix row referred to by the given sudoku coordinates.
 * @param row 1-based row in the sudoku
 * @param col 1-based column in the sudoku
 * @param num number in the sudoku
 * @return 0-based row index in the exact cover matrix.
 */
size_t
getRowIndex(size_t row, size_t col, int num)
{
    return (row - 1) * sudokuSide * sudokuSide + (col - 1) * sudokuSide + (num - 1);
}

/**
 * Creates a matrix representing the exact cover problem of an empty sudoku.
 * @return
 */
DlxConstraintMatrix
createEmptyECMatrix()
{
    size_t rows = 9 * 9 * 9; // sudoku_rows * sudoku_cols * sudoku_nums
    size_t cols = 9 * 9 * 4; // sudoku_rows * sudoku_nums for each constraint type

    // zero-initializes the matrix of size [rows, cols]
    DlxConstraintMatrix matrix(rows, cols);

    size_t nextColumn = 0;

    // Each row in the matrix refers to a specific number in a specific position. This is encoded by the first block.
    for (size_t row = 1; row <= sudokuSide; row++) {
        for (size_t col = 1; col <= sudokuSide; col++) {
            for (int num = 1; num <= sudokuSide; num++) {
                matrix.constraints[getRowIndex(row, col, num)].push_back(nextColumn);
            }
            nextColumn++;
        }
    }

    // Constraints to ensure that each number only occurs once per row
    for (size_t row = 1; row <= sudokuSide; row++) {
        for (int num = 1; num <= sudokuSide; num++) {
            for (size_t col = 1; col <= sudokuSide; col++) {
                matrix.constraints[getRowIndex(row, col, num)].push_back(nextColumn);
            }
            nextColumn++;
        }
    }

    // Constraints to ensure that each number only occurs once per column
    for (size_t col = 1; col <= sudokuSide; col++) {
        for (int num = 1; num <= sudokuSide; num++) {
            for (size_t row = 1; row <= sudokuSide; row++) {
                matrix.constraints[getRowIndex(row, col, num)].push_back(nextColumn);
            }
            nextColumn++;
        }
    }

    // Constraints to ensure that each number only occurs once per block
    for (size_t boxRow = 1; boxRow <= sudokuSide; boxRow += boxSide) {
        for (size_t boxCol = 1; boxCol <= sudokuSide; boxCol += boxSide) {
            for (int num = 1; num <= sudokuSide; num++) {
                for (size_t rowDelta = 0; rowDelta < boxSide; rowDelta++) {
                    for (size_t colDelta = 0; colDelta < boxSide; colDelta++) {
                        matrix.constraints[getRowIndex(boxRow + rowDelta, boxCol + colDelta, num)].push_back(
                          nextColumn);
                    }
                }
                nextColumn++;
            }
        }
    }

    return matrix;
}

int
atGrid(const SudokuGrid& grid, size_t row, size_t col)
{
    return grid[row * sudokuSide + col];
}

DlxConstraintMatrix
createSudokuECMatrix(SudokuGrid sudoku)
{
    DlxConstraintMatrix matrix = createEmptyECMatrix();
    for (size_t i = 1; i <= sudokuSide; i++) {
        for (size_t j = 1; j <= sudokuSide; j++) {
            int sudokuNum = atGrid(sudoku, i - 1, j - 1);

            if (sudokuNum == 0) {
                continue;
            }

            // zero out in the constraint board
            for (int num = 1; num <= sudokuSide; num++) {
                if (num != sudokuNum) {
                    auto& row = matrix.constraints[getRowIndex(i, j, num)];
                    row.clear();
                }
            }
        }
    }
    return matrix;
}

class SudokuDlxSolver : public SudokuSolver
{
  public:
    ~SudokuDlxSolver() override = default;

    [[nodiscard]] std::unique_ptr<SudokuGrid> solve(const SudokuGrid& sudoku) const override;
};

std::unique_ptr<SudokuGrid>
SudokuDlxSolver::solve(const SudokuGrid& sudoku) const
{
    auto sudokuGrid = createSudokuECMatrix(sudoku);

    DlxSolver solver;
    auto resultRows = solver.solve(sudokuGrid);

    if (resultRows == nullptr) {
        return nullptr;
    }

    auto solution = std::make_unique<SudokuGrid>();
    for (auto row : resultRows->constraints) {
        // Use leftmost node to decode position in sudoku.
        int lmIndex = static_cast<int>(row.at(0));
        int r = lmIndex / 9;
        int c = lmIndex % 9;

        // The next neighbor of the leftmost node encodes the row/number. -> Use it to decode the number.
        int num = (static_cast<int>(row.at(1)) % 9) + 1;

        (*solution)[r * sudokuSide + c] = num;
    }
    return solution;
}

std::unique_ptr<SudokuSolver>
SudokuSolver::create(SolverType type)
{
    switch (type) {
        case SolverType::Constraint:
            return std::make_unique<SudokuConstraintSolver>();
        case SolverType::Dlx:
            return std::make_unique<SudokuDlxSolver>();
        default:
            throw std::runtime_error("Invalid solver type.");
    }
}
