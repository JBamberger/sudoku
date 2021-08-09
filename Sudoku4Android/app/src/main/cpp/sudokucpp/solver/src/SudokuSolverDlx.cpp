//
// Created by jannik on 08.08.2021.
//

#include "SudokuSolverDlx.h"

#include "SudokuSolver.h"
#include "dlx.h"
#include <vector>

static int sudokuSide = 9;
static int boxSide = 3;

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
std::vector<std::vector<int>>
sudokuExactCover()
{
    size_t rows = 9 * 9 * 9; // sudoku_rows * sudoku_cols * sudoku_nums
    size_t cols = 9 * 9 * 4; // sudoku_rows * sudoku_nums for each constraint type

    // zero-initializes the matrix of size [rows, cols]
    std::vector<std::vector<int>> matrix(rows, std::vector<int>(cols));

    size_t nextColumn = 0;

    // Each row in the matrix refers to a specific number in a specific position. This is encoded by the first block.
    for (size_t row = 1; row <= sudokuSide; row++) {
        for (size_t col = 1; col <= sudokuSide; col++) {
            for (int num = 1; num <= sudokuSide; num++) {
                matrix[getRowIndex(row, col, num)][nextColumn] = 1;
            }
            nextColumn++;
        }
    }

    // Constraints to ensure that each number only occurs once per row
    for (size_t row = 1; row <= sudokuSide; row++) {
        for (int num = 1; num <= sudokuSide; num++) {
            for (size_t col = 1; col <= sudokuSide; col++) {
                matrix[getRowIndex(row, col, num)][nextColumn] = 1;
            }
            nextColumn++;
        }
    }

    // Constraints to ensure that each number only occurs once per column
    for (size_t col = 1; col <= sudokuSide; col++) {
        for (int num = 1; num <= sudokuSide; num++) {
            for (size_t row = 1; row <= sudokuSide; row++) {
                matrix[getRowIndex(row, col, num)][nextColumn] = 1;
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
                        matrix[getRowIndex(boxRow + rowDelta, boxCol + colDelta, num)][nextColumn] = 1;
                    }
                }
                nextColumn++;
            }
        }
    }

    return matrix;
}

int
main()
{
    auto emptyGrid = sudokuExactCover();

    DlxSolver solver;
    solver.solve(emptyGrid);

    //    for (const auto& row : emptyGrid) {
    //        for (const auto& val : row) {
    //            std::cout << ' ' << val;
    //        }
    //        std::cout << std::endl;
    //    }
}
