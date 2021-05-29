#ifndef SUDOKUSOLVER_H
#define SUDOKUSOLVER_H

#include <array>
#include <iostream>
#include <memory>
#include <string>

using SudokuGrid = std::array<int, 81>;

class SudokuSolver
{
    template<int gridSize>
    struct Impl;
    std::unique_ptr<Impl<9>> pimpl;

  public:
    explicit SudokuSolver();
    ~SudokuSolver();
    SudokuSolver(SudokuSolver&&) noexcept;
    SudokuSolver(const SudokuSolver&) = delete;
    SudokuSolver& operator=(SudokuSolver&&) noexcept;
    SudokuSolver& operator=(const SudokuSolver&) = delete;

    std::unique_ptr<SudokuGrid> solve(const SudokuGrid& sudoku) const;
};

inline void
printGrid(const SudokuGrid& grid, bool flat = true)
{
    for (int i = 0; i < grid.size(); i++) {
        std::cout << " " << (grid[i] == 0 ? "." : std::to_string(grid[i]));
        if (flat) {
            if (i % 9 == 8) {
                std::cout << "|";
            }
        } else {

            if (i % 9 == 8) {
                std::cout << "\n";
            } else if (i % 3 == 2) {
                std::cout << " |";
            }

            if (i % 27 == 26 && i != 80) {
                std::cout << " ------+-------+------\n";
            }
        }
    }

    std::cout << std::endl;
}

#endif // SUDOKUSOLVER_H
