#include "SudokuSolver.hpp"

#include <assert.h>

/**
 * Slow square root algorithm which computes the root of a number.
 */
static size_t
sqrt(size_t x)
{
    assert(x >= 0);

    if (x == 0 || x == 1)
        return x;
    size_t i = 1;
    size_t result = 1;
    while (result <= x) {
        i++;
        result = i * i;
    }
    return i - 1;
}

static bool
is_valid_value(std::vector<std::vector<int>> board, size_t row, size_t col, int value)
{
    const size_t S = board.size();
    const size_t block_size = sqrt(S);

    const size_t row_start = row - row % block_size;
    const size_t col_start = col - col % block_size;

    for (size_t i = 0; i < S; i++) {
        if (board[row][i] == value)
            return false; // check row
        if (board[i][col] == value)
            return false; // check col
    }
    for (size_t r = 0; r < block_size; r++) {
        for (size_t c = 0; c < block_size; c++) {
            if (board[row_start + r][col_start + c] == value)
                return false; // check block
        }
    }
    return true;
}

static void
solveImpl(std::vector<std::vector<int>> board, size_t pos)
{
    const size_t S = board.size();

    if (pos == S * S) {
        return; // the board is completed
    } else {
        size_t row = pos / S;
        size_t col = pos % S;

        if (board[row][col] > 0) {
            solveImpl(board,
                      pos + 1); // the field is already occupied, go to next one
        } else {
            for (int i = 1; i <= 9; i++) {
                if (is_valid_value(board, row, col, i)) {
                    board[row][col] = i;
                    solveImpl(board, pos + 1);
                    board[row][col] = 0; // rollback change
                }
            }
        }
        throw std::exception("There exists no solution for this Sudoku."); // there is no solution to
                                                                           // the sudoku
    }
}

void
solve(std::vector<std::vector<int>>& board)
{
    solveImpl(board, 1);
}
