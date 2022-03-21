
#ifndef SUDOKU4ANDROID_DLXSUDOKUSOLVER_H
#define SUDOKU4ANDROID_DLXSUDOKUSOLVER_H

#include "SudokuSolver.h"

class SudokuDlxSolver : public SudokuSolver
{
  public:
    ~SudokuDlxSolver() override = default;

    [[nodiscard]] std::unique_ptr<SudokuGrid> solve(const SudokuGrid& sudoku) const override;
};

#endif // SUDOKU4ANDROID_DLXSUDOKUSOLVER_H
