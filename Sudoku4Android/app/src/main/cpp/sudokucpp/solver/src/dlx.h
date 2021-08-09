//
// Created by jannik on 08.08.2021.
//

#ifndef DLX_H
#define DLX_H

#include <functional>
#include <memory>
#include <vector>

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

    void solve(const std::vector<std::vector<int>>& constraintMatrix,
               const std::function<void(std::unique_ptr<std::vector<std::vector<size_t>>>)>& resultCollector);
    std::unique_ptr<std::vector<std::vector<size_t>>> solve(const std::vector<std::vector<int>>& constraintMatrix);
};

#endif // DLX_H
