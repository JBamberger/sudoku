//
// Created by jannik on 08.08.2021.
//

#ifndef DLX_H
#define DLX_H

#include <vector>
#include <memory>

class DlxSolver {
    struct Impl;
    std::unique_ptr<Impl> impl;

  public:
    explicit DlxSolver();
    ~DlxSolver();
    DlxSolver(DlxSolver&&) noexcept;
    DlxSolver(const DlxSolver&) = delete;
    DlxSolver& operator=(DlxSolver&&) noexcept;
    DlxSolver& operator=(const DlxSolver&) = delete;

    void solve(const std::vector<std::vector<int>>& constraintMatrix);

};

#endif // DLX_H
