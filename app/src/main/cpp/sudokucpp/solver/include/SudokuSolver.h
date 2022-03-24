#ifndef SUDOKUSOLVER_H
#define SUDOKUSOLVER_H

#include "sudoku_grid.h"
#include <cassert>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

enum class SolverType
{
    /**
     * This solver uses a backtracking algorithm to solve Sudokus. To improve the performance some simple heuristics
     * and the simplifying of a small set of known situations is used.
     */
    Constraint,
    /**
     * This algorithm is based on Knuth's Algorithm X with the 'Dancing Links' implementation.
     */
    Dlx
};

/**
 * Interface for a ConstraintSolverSudoku solving algorithm.
 */
class SudokuSolver
{
  public:
    virtual ~SudokuSolver() = default;
    /**
     * Solves the given sudoku grid.
     * @param sudoku input sudoku grid
     * @return Pointer to the solution or nullptr if there is no solution to the grid.
     */
    [[nodiscard]] virtual std::unique_ptr<SudokuGrid> solve(const SudokuGrid& sudoku) const = 0;

    static std::unique_ptr<SudokuSolver> create(SolverType type = SolverType::Dlx);
};

#endif // SUDOKUSOLVER_H
