#ifndef SUDOKU4ANDROID_CONSTRAINTSUDOKUSOLVER_H
#define SUDOKU4ANDROID_CONSTRAINTSUDOKUSOLVER_H

#include "SudokuSolver.h"

class SudokuConstraintSolver : public SudokuSolver
{
  public:
    ~SudokuConstraintSolver() override;

    [[nodiscard]] std::unique_ptr<SudokuGrid> solve(const SudokuGrid& sudokuGrid) const override;
};

#endif // SUDOKU4ANDROID_CONSTRAINTSUDOKUSOLVER_H
