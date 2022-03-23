
#include "DlxSudokuSolver.h"
#include "dlx.h"

static size_t sudokuSide = 9;

DlxConstraintMatrix
createSudokuECMatrix(const SudokuGrid& sudoku)
{

    const size_t S = sudoku.sideLen;
    const size_t rows = S * S * S; // sudoku_rows * sudoku_cols * sudoku_nums
    const size_t cols = S * S * 4; // sudoku_rows * sudoku_nums for each constraint type

    /**
     * Computes the exact cover matrix row referred to by the given sudoku coordinates.
     * @param row 1-based row in the sudoku
     * @param col 1-based column in the sudoku
     * @param num number in the sudoku
     * @return 0-based row index in the exact cover matrix.
     */
    auto getRowIndex = [S](size_t row, size_t col, int num) { return (row - 1) * S * S + (col - 1) * S + (num - 1); };

    // zero-initializes the matrix of size [rows, cols]
    DlxConstraintMatrix matrix(rows, cols);

    size_t nextColumn = 0;

    // Each row in the matrix refers to a specific number in a specific position. This is encoded by the first block.
    for (size_t row = 1; row <= S; row++) {
        for (size_t col = 1; col <= S; col++) {
            for (int num = 1; num <= S; num++) {
                matrix.constraints[getRowIndex(row, col, num)].push_back(nextColumn);
            }
            nextColumn++;
        }
    }

    // Constraints to ensure that each number only occurs once per row
    for (size_t row = 1; row <= S; row++) {
        for (int num = 1; num <= S; num++) {
            for (size_t col = 1; col <= S; col++) {
                matrix.constraints[getRowIndex(row, col, num)].push_back(nextColumn);
            }
            nextColumn++;
        }
    }

    // Constraints to ensure that each number only occurs once per column
    for (size_t col = 1; col <= S; col++) {
        for (int num = 1; num <= S; num++) {
            for (size_t row = 1; row <= S; row++) {
                matrix.constraints[getRowIndex(row, col, num)].push_back(nextColumn);
            }
            nextColumn++;
        }
    }

    // Constraints to ensure that each number only occurs once per block
    for (size_t boxRow = 1; boxRow <= S; boxRow += sudoku.boxLen) {
        for (size_t boxCol = 1; boxCol <= S; boxCol += sudoku.boxLen) {
            for (int num = 1; num <= S; num++) {
                for (size_t rowDelta = 0; rowDelta < sudoku.boxLen; rowDelta++) {
                    for (size_t colDelta = 0; colDelta < sudoku.boxLen; colDelta++) {
                        matrix.constraints[getRowIndex(boxRow + rowDelta, boxCol + colDelta, num)].push_back(
                          nextColumn);
                    }
                }
                nextColumn++;
            }
        }
    }

    // Fill with actual sudoku data by removing impossible rows
    for (size_t i = 1; i <= sudoku.sideLen; i++) {
        for (size_t j = 1; j <= sudoku.sideLen; j++) {
            int sudokuNum = sudoku.at(i - 1, j - 1);

            // cell is empty, do nothing
            if (sudokuNum == 0)
                continue;

            // zero out in the constraint board
            for (int num = 1; num <= sudoku.sideLen; num++) {
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
    // Matrix rows describe the blocked elements by setting a specific row/col to a specific number
    // The columns describe various constraints that must hold.
    // [  0 -  80]: Each cell is used only once/by one number
    // [ 81 - 161]: Each row contains each number
    // [162 - 242]: Each column contains each number
    // [243 - 323]: Each box contains each number

    const size_t S = sudoku.sideLen;
    const size_t B = sudoku.boxLen;

    // zero-initializes the matrix of size [rows, cols]
    DlxConstraintMatrix matrix(S * S * S, S * S * 4);

    auto addConstraints = [S, B, &matrix](int row, int col, int num) {
        const size_t SS = S * S;

        auto& constraint = matrix.constraints.at(SS * row + S * col + num);
        constraint.reserve(4);

        // One number per cell
        constraint.emplace_back((S * row + col));
        // Each number once per row
        constraint.emplace_back(SS + (S * row + num));
        // Each number once per col
        constraint.emplace_back(SS + SS + (S * col + num));
        // Each number once per box
        constraint.emplace_back(SS + SS + SS + (S * (B * (row / B) + (col / B)) + num));
    };

    for (int row = 0; row < S; row++) {
        for (int col = 0; col < S; col++) {
            int sudokuNum = sudoku[row * S + col];

            if (sudokuNum <= 0) {
                // The cell is empty, add all possible placements in this cell
                for (int num = 0; num < S; num++) {
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

    auto solution = std::make_unique<SudokuGrid>(sudoku.sideLen);
    for (auto row : resultRows->constraints) {
        size_t i1 = row.at(0);
        size_t i2 = row.at(1);

        // Use leftmost node to decode position in sudoku.
        size_t r = i1 / sudoku.sideLen;
        size_t c = i1 % sudoku.sideLen;

        // The next neighbor of the leftmost node encodes the row/number. -> Use it to decode the number.
        cell_t num = static_cast<cell_t>(i2 % sudoku.sideLen) + 1;

        solution->at(r, c) = num;
    }
    return solution;
}
