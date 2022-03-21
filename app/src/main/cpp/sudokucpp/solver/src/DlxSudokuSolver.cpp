
#include "DlxSudokuSolver.h"
#include "dlx.h"

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

int
atGrid(const SudokuGrid& grid, size_t row, size_t col)
{
    return grid[row * sudokuSide + col];
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

DlxConstraintMatrix
createSudokuECMatrix(const SudokuGrid& sudoku)
{
    DlxConstraintMatrix matrix = createEmptyECMatrix();
    for (size_t i = 1; i <= sudokuSide; i++) {
        for (size_t j = 1; j <= sudokuSide; j++) {
            int sudokuNum = atGrid(sudoku, i - 1, j - 1);

            // cell is empty, do nothing
            if (sudokuNum == 0)
                continue;

            // zero out in the constraint board
            for (int num = 1; num <= sudokuSide; num++) {
                if (num == sudokuNum)
                    continue;

                matrix.constraints.at(getRowIndex(i, j, num)).clear();
            }
        }
    }
    return matrix;
}

DlxConstraintMatrix
createSudokuECMatrix_v2(const SudokuGrid& sudoku)
{
    // Constraint matrix for a sudoku.
    // Size: 9*9*9 x 9*9*4 == 729 x 324

    // Matrix rows describe the blocked elements by setting a specific row/col to a specific number
    // The columns describe various constraints that must hold.
    // [  0 -  80]: Each cell is used only once/by one number
    // [ 81 - 161]: Each row contains each number
    // [162 - 242]: Each column contains each number
    // [243 - 323]: Each box contains each number

    size_t rows = 9 * 9 * 9; // All cell+number combinations
    size_t cols = 9 * 9 * 4; // Constraints for cell placement, row, col and block per position

    // zero-initializes the matrix of size [rows, cols]
    DlxConstraintMatrix matrix(rows, cols);

    auto addConstraints = [&matrix](int row, int col, int num) {
        const size_t N = sudokuSide;
        const size_t B = boxSide;
        const size_t NN = N * N;
        auto& constraint = matrix.constraints.at(NN * row + N * col + num);

        // One number per cell
        constraint.emplace_back((N * row + col));
        // Each number once per row
        constraint.emplace_back(NN + (N * row + num));
        // Each number once per col
        constraint.emplace_back(NN + NN + (N * col + num));
        // Each number once per box
        constraint.emplace_back(NN + NN + NN + (N * (B * (row / B) + (col / B)) + num));
    };

    for (int row = 0; row < sudokuSide; row++) {
        for (int col = 0; col < sudokuSide; col++) {
            int sudokuNum = sudoku[row * sudokuSide + col];

            if (sudokuNum <= 0) {
                // The cell is empty, add all possible placements in this cell
                for (int num = 0; num < sudokuSide; num++) {
                    addConstraints(row, col, num);
                }
            } else {
                // There is already a number in the grid, only remove all possibilities to place a different number in
                // this cell by adding only one set of constraints.
                addConstraints(row, col, sudokuNum - 1);
            }
        }
    }

    return matrix;
}

std::unique_ptr<SudokuGrid>
SudokuDlxSolver::solve(const SudokuGrid& sudoku) const
{
    auto sudokuGrid = createSudokuECMatrix_v2(sudoku);

    DlxSolver solver;
    auto resultRows = solver.solve(sudokuGrid);

    if (resultRows == nullptr) {
        return nullptr;
    }

    auto solution = std::make_unique<SudokuGrid>();
    for (auto row : resultRows->constraints) {
        int i1 = static_cast<int>(row.at(0));
        int i2 = static_cast<int>(row.at(1));

        // Use leftmost node to decode position in sudoku.
        int r = i1 / 9;
        int c = i1 % 9;

        // The next neighbor of the leftmost node encodes the row/number. -> Use it to decode the number.
        int num = (i2 % 9) + 1;

        (*solution)[r * sudokuSide + c] = num;
    }
    return solution;
}
