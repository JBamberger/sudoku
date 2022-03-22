
#include <benchmark/benchmark.h>

#include "SudokuSolver.h"
#include "dlx.h"

DlxConstraintMatrix
createSudokuECMatrix(const SudokuGrid& sudoku);

DlxConstraintMatrix
createSudokuECMatrix_v2(const SudokuGrid& sudoku);

static void
BM_InitECMatV1(benchmark::State& state)
{
    SudokuGrid sudoku{};
    std::fill(sudoku.begin(), sudoku.end(), 0);
    volatile size_t x;
    for (auto _ : state) {
        x = createSudokuECMatrix(sudoku).constraints.size();
    }
}
BENCHMARK(BM_InitECMatV1);

static void
BM_InitECMatV2(benchmark::State& state)
{
    SudokuGrid sudoku{};
    std::fill(sudoku.begin(), sudoku.end(), 0);
    volatile size_t x;
    for (auto _ : state) {
        x = createSudokuECMatrix_v2(sudoku).constraints.size();
    }
}
BENCHMARK(BM_InitECMatV2);

BENCHMARK_MAIN();