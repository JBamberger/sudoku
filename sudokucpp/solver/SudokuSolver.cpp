#include "SudokuSolver.h"

#include <algorithm>
#include <numeric>
#include <set>

struct Sudoku
{
    static constexpr int BOX_SIZE = 3;
    static constexpr int SIZE = 9;

    SudokuGrid grid;
    std::array<std::set<int>, 81> candidates;

    static int from2d(int row, int col) { return row * SIZE + col; }

    explicit Sudoku(const SudokuGrid& grid)
      : grid(grid)
      , candidates()
    {
        for (int i = 0; i < SIZE; i++) {
            for (int j = 0; j < SIZE; j++) {
                candidates[from2d(i, j)] = findCandidates(i, j);
            }
        }
    }

    std::set<int> findCandidates(int r, int c)
    {
        if (grid[from2d(r, c)] != 0) {
            return std::set<int>();
        }
        std::set<int> options;

        return options;
    }
};

template<int GRID_SIZE>
struct SudokuSolver::Impl
{
    void hasConstraintViolations(const SudokuGrid& sudoku) const {}

    void solveTrivialCells(Sudoku& sudoku) const {}

    std::unique_ptr<SudokuGrid> solve(Sudoku& sudoku, int depth)
    {
        while (true) {
            bool edited = false;
            bool solved = true;

            for (int i = 0; i < GRID_SIZE; i++) {
                for (int j = 0; j < GRID_SIZE; j++) {
                    if (sudoku.)
                }
            }
        }
    }
};

SudokuSolver::SudokuSolver()
  : pimpl{ std::make_unique<Impl<9>>() }
{}
SudokuSolver&
SudokuSolver::operator=(SudokuSolver&&) = default;
SudokuSolver::SudokuSolver(SudokuSolver&&) = default;
SudokuSolver::~SudokuSolver() = default;

std::unique_ptr<SudokuGrid>
SudokuSolver::solve(const SudokuGrid& sudoku) const
{
    const auto& msg = pimpl->hasConstraintViolations(sudoku);

    if (msg == nullptr) {
        return std::unique_ptr<SudokuGrid>();
    }

    pimpl->solveTrivialCells(sudoku);
    return pimpl->solve(sudoku, 0);
}
