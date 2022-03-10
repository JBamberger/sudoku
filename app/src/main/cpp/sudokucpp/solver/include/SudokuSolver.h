#ifndef SUDOKUSOLVER_H
#define SUDOKUSOLVER_H

#include <array>
#include <iostream>
#include <memory>
#include <string>

using SudokuGrid = std::array<int, 81>;

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
