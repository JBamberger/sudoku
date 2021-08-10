//
// Created by jannik on 08.08.2021.
//

#include "SudokuSolverDlx.h"

#include "SudokuSolver.h"
#include "dlx.h"
#include <vector>
#include <fstream>

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
std::vector<std::vector<int>>
createEmptyECMatrix()
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
atGrid(const SudokuGrid& grid, size_t row, size_t col)
{
    return grid[row * sudokuSide + col];
}

std::vector<std::vector<int>>
createSudokuECMatrix(SudokuGrid sudoku)
{
    std::vector<std::vector<int>> matrix = createEmptyECMatrix();
    for (size_t i = 1; i <= sudokuSide; i++) {
        for (size_t j = 1; j <= sudokuSide; j++) {
            int sudokuNum = atGrid(sudoku, i - 1, j - 1);

            if (sudokuNum == 0) {
                continue;
            }

            // zero out in the constraint board
            for (int num = 1; num <= sudokuSide; num++) {
                if (num != sudokuNum) {
                    auto& row = matrix[getRowIndex(i, j, num)];
                    std::fill(std::begin(row), std::end(row), 0);
                }
            }
        }
    }
    return matrix;
}

int
main()
{

    std::string sudokuString = "4.....8.5.3..........7......2.....6.....8.4......1.......6.3.7.5..2.....1.4......";
    SudokuGrid grid{};
    for (int i = 0; i < 81; i++) {
        char cellValue = sudokuString.at(i);
        grid[i] = cellValue == '.' ? 0 : cellValue - '0';
    }
    auto emptyGrid = createEmptyECMatrix();
    auto sudokuGrid = createSudokuECMatrix(grid);


    std::ofstream emptyGridFile("emptyGrid.txt");
    for (const auto& row : emptyGrid) {
        for (auto value : row) {
            emptyGridFile << ' ' << value;
        }
        emptyGridFile << std::endl;
    }
    emptyGridFile.close();

    std::ofstream sudokuGridFile("sudokuGrid.txt");
    for (const auto& row : sudokuGrid) {
        for (auto value : row) {
            sudokuGridFile << ' ' << value;
        }
        sudokuGridFile << std::endl;
    }
    sudokuGridFile.close();


    DlxSolver solver;
    auto resultRows = solver.solve(sudokuGrid);

    if (resultRows == nullptr) {
        std::cerr << "Could not find any solutions to the sudoku." << std::endl;
    } else {
        std::array<std::array<int, 9>, 9> result{};
        SudokuGrid solution{};
        for (auto row : *resultRows) {
            // Use leftmost node to decode position in sudoku.
            int lmIndex = static_cast<int>(row.at(0));
            int r = lmIndex / 9;
            int c = lmIndex % 9;

            // The next neighbor of the leftmost node encodes the row/number. -> Use it to decode the number.
            int num = (static_cast<int>(row.at(1)) % 9) + 1;

            result[r][c] = num;

            auto coord = r * sudokuSide + c;
            std::cout << coord << std::endl;
            solution[coord] = num;
        }

        for (auto row : result) {
            for (auto value : row) {
                std::cout << ' ' << value;
            }
            std::cout << std::endl;
        }

        for (int i = 0; i < 81; i++) {
            std::cout << solution[i] ;
        }
        std::cout << std::endl;
    }
}
