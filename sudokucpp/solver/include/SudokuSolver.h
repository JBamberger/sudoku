#ifndef SUDOKUSOLVER_H
#define SUDOKUSOLVER_H

#include <array>
#include <memory>

using SudokuGrid = std::array<int, 81>;

class SudokuSolver
{
    template<int gridSize>
    struct Impl;
    std::unique_ptr<Impl<9>> pimpl;
  public:
    explicit SudokuSolver();
    ~SudokuSolver();
    SudokuSolver(SudokuSolver&&) noexcept ;
    SudokuSolver(const SudokuSolver&) = delete;
    SudokuSolver& operator=(SudokuSolver&&) noexcept ;
    SudokuSolver& operator=(const SudokuSolver&) = delete;


    std::unique_ptr<SudokuGrid> solve(const SudokuGrid& sudoku) const;
};

#endif // SUDOKUSOLVER_H
