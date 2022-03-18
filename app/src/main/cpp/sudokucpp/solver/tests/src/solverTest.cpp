#include "gtest/gtest.h"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

#include <SudokuSolver.h>
#include <test_helper.h>

namespace fs = std::filesystem;

namespace {

void
runTest(std::unique_ptr<SudokuSolver> solver, const std::string& path)
{
    const auto challenges = readSudokuChallenges(path);

    for (const auto& challenge : challenges) {
        const auto result = solver->solve(challenge.grid);
        EXPECT_NE(result, nullptr);

        if (result) {
            int errCnt = 0;
            if (!challenge.isValidSolution(result.get(), &errCnt)) {
                std::cout << "Failed with " << errCnt << " errors." << std::endl;
                printGrid(*result, false);

                printGrid(*result);
                printGrid(challenge.solution);
                printGrid(challenge.grid);
            }
            EXPECT_EQ(errCnt, 0);
        }
    }
}

TEST(solver, test_test_sudokus)
{
    runTest(SudokuSolver::create(SolverType::Dlx), "./solver/tests/test_sudokus.txt");
}

TEST(solver, test_any_sudokus)
{
    runTest(SudokuSolver::create(SolverType::Dlx), "./solver/tests/any.txt");
}

TEST(solver, test_simple_sudokus)
{
    runTest(SudokuSolver::create(SolverType::Dlx), "./solver/tests/simple.txt");
}

TEST(solver, test_easy_sudokus)
{
    runTest(SudokuSolver::create(SolverType::Dlx), "./solver/tests/easy.txt");
}

TEST(solver, test_intermediate_sudokus)
{
    runTest(SudokuSolver::create(SolverType::Dlx), "./solver/tests/intermediate.txt");
}

TEST(solver, test_expert_sudokus)
{
    runTest(SudokuSolver::create(SolverType::Dlx), "./solver/tests/expert.txt");
}
}
