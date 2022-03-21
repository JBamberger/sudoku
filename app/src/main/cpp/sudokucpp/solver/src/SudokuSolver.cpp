
#include "SudokuSolver.h"
#include "ConstraintSudokuSolver.h"
#include "DlxSudokuSolver.h"

#include <stdexcept>

std::unique_ptr<SudokuSolver>
SudokuSolver::create(SolverType type)
{
    switch (type) {
        case SolverType::Constraint:
            return std::make_unique<SudokuConstraintSolver>();
        case SolverType::Dlx:
            return std::make_unique<SudokuDlxSolver>();
        default:
            throw std::runtime_error("Invalid solver type.");
    }
}
