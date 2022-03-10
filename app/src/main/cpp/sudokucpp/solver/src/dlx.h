//
// Created by jannik on 08.08.2021.
//

#ifndef DLX_H
#define DLX_H

#include <functional>
#include <memory>
#include <vector>

struct DlxConstraintMatrix
{
    size_t numCols;
    std::vector<std::vector<size_t>> constraints;

    DlxConstraintMatrix(size_t numRows, size_t numCols)
      : numCols(numCols)
      , constraints(numRows)
    {}
};

class DlxSolver
{
    struct Impl;
    std::unique_ptr<Impl> impl;

  public:
    explicit DlxSolver();
    ~DlxSolver();
    DlxSolver(DlxSolver&&) noexcept;
    DlxSolver(const DlxSolver&) = delete;
    DlxSolver& operator=(DlxSolver&&) noexcept;
    DlxSolver& operator=(const DlxSolver&) = delete;

    void solve(const DlxConstraintMatrix& constraintMatrix,
               const std::function<void(std::unique_ptr<DlxConstraintMatrix>)>& resultCollector);
    std::unique_ptr<DlxConstraintMatrix> solve(const DlxConstraintMatrix& constraintMatrix);
};

#endif // DLX_H
