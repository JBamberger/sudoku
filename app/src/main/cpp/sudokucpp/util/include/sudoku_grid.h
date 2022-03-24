#ifndef SUDOKU4ANDROID_SUDOKU_GRID_H
#define SUDOKU4ANDROID_SUDOKU_GRID_H

#include <iostream>
#include <vector>

using cell_t = int;

class SudokuGrid
{
    std::vector<cell_t> elements;

  public:
    const size_t sideLen;
    const size_t boxLen;

    explicit SudokuGrid(size_t side_len);

    SudokuGrid(const SudokuGrid& other) = default;

    [[nodiscard]] cell_t& operator[](size_t idx);
    [[nodiscard]] const cell_t& operator[](size_t idx) const;

    [[nodiscard]] cell_t& at(size_t idx);
    [[nodiscard]] const cell_t& at(size_t idx) const;

    [[nodiscard]] cell_t& at(size_t row, size_t col);
    [[nodiscard]] const cell_t& at(size_t row, size_t col) const;

    [[nodiscard]] size_t size() const noexcept;

    [[nodiscard]] bool isCellFilled(size_t row, size_t col) const;

    static void printGrid(std::ostream& os, const SudokuGrid& grid, bool flat = true);

    friend std::ostream& operator<<(std::ostream& output, const SudokuGrid& grid);
};


#endif // SUDOKU4ANDROID_SUDOKU_GRID_H
