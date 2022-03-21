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

/**
 * DLX implementation of Algorithm X (c.f. https://arxiv.org/abs/cs/0011047)
 *
 * The algorithm solves the Exact Cover Problem:
 * Selects a subset of rows from a binary matrix such that each columns contains
 * exactly a single 1.
 */
class DlxSolver
{
    class Impl;
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
