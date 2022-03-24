#include "sudoku_grid.h"
#include <cassert>
#include <memory>
#include <sstream>
#include <string>

SudokuGrid::SudokuGrid(size_t side_len)
  : sideLen(side_len)
  , boxLen(static_cast<size_t>(std::sqrt(side_len)))
  , elements(side_len * side_len)
{
    assert(boxLen * boxLen == side_len); // The side length must be a perfect square for a well-defined sudoku
}

cell_t&
SudokuGrid::operator[](size_t idx)
{
    return elements[idx];
}

const cell_t&
SudokuGrid::operator[](size_t idx) const
{
    return elements[idx];
}

cell_t&
SudokuGrid::at(size_t idx)
{
    return elements.at(idx);
}

const cell_t&
SudokuGrid::at(size_t idx) const
{
    return elements.at(idx);
}

cell_t&
SudokuGrid::at(size_t row, size_t col)
{
    assert(row < sideLen && col < sideLen);

    return elements[sideLen * row + col];
}

const cell_t&
SudokuGrid::at(size_t row, size_t col) const
{
    assert(row < sideLen && col < sideLen);

    return elements[sideLen * row + col];
}

bool
SudokuGrid::isCellFilled(size_t row, size_t col) const
{
    return at(row, col) > 0;
}

size_t
SudokuGrid::size() const noexcept
{
    return elements.size();
}

void
SudokuGrid::printGrid(std::ostream& os, const SudokuGrid& grid, bool flat)
{
    std::string vblock_div;
    if (!flat) {
        std::stringstream ss;
        for (size_t i = 0; i < grid.boxLen; i++) {
            for (size_t j = 0; j < grid.boxLen; j++) {
                ss << '-';
            }
            if (i != grid.boxLen - 1) {
                ss << '+';
            }
        }
        ss << '\n';
        vblock_div = ss.str();
    }

    for (size_t row = 0; row < grid.sideLen; row++) {
        for (size_t col = 0; col < grid.sideLen; col++) {
            auto val = grid.at(row, col);
            os << (val == 0 ? "." : std::to_string(val));
            if (!flat && (col % grid.boxLen == grid.boxLen - 1)) {

                os << ((col != grid.sideLen - 1) ? '|' : '\n');
            }
        }
        if (!flat && (row != grid.sideLen - 1) && (row % grid.boxLen == grid.boxLen - 1)) {
            os << vblock_div;
        }
    }
}

std::ostream&
operator<<(std::ostream& output, const SudokuGrid& grid)
{
    SudokuGrid::printGrid(output, grid, true);
    return output;
}