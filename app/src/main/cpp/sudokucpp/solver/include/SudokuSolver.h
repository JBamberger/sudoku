#ifndef SUDOKUSOLVER_H
#define SUDOKUSOLVER_H

#include <cassert>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

using cell_t = int;

class SudokuGrid
{
    std::vector<cell_t> elements;

  public:
    const size_t sideLen;
    const size_t boxLen;

    explicit inline SudokuGrid(size_t side_len)
      : sideLen(side_len)
      , boxLen(static_cast<size_t>(std::sqrt(side_len)))
      , elements(side_len * side_len)
    {
        assert(boxLen * boxLen == side_len); // The side length must be a perfect square for a well-defined sudoku
    }

    SudokuGrid(const SudokuGrid& other) = default;

    [[nodiscard]] inline cell_t& operator[](size_t idx) { return elements[idx]; }
    [[nodiscard]] inline const cell_t& operator[](size_t idx) const { return elements[idx]; }

    [[nodiscard]] inline cell_t& at(size_t idx) { return elements.at(idx); }
    [[nodiscard]] inline const cell_t& at(size_t idx) const { return elements.at(idx); }

    [[nodiscard]] inline cell_t& at(size_t row, size_t col)
    {
        assert(row < sideLen && col < sideLen);

        return elements[sideLen * row + col];
    }
    [[nodiscard]] inline const cell_t& at(size_t row, size_t col) const
    {
        assert(row < sideLen && col < sideLen);

        return elements[sideLen * row + col];
    }

    [[nodiscard]] inline size_t size() const noexcept { return elements.size(); }

    [[nodiscard]] inline bool isCellFilled(int row, int col) const { return at(row, col) > 0; }

    friend inline std::ostream& operator<<(std::ostream& output, const SudokuGrid& grid)
    {
        for (size_t row = 0; row < grid.sideLen; row++) {
            for (size_t col = 0; col < grid.sideLen; col++) {
                cell_t cell = grid.at(row, col);
                output << " " << (cell == 0 ? "." : std::to_string(cell));
            }
            if (row != grid.sideLen - 1) {
                output << "|";
            }
        }
        output << "\n";
        return output;
    }
};

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
