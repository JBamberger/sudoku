#include "gtest/gtest.h"

#include <fstream>
#include <iostream>
#include <string>

#include <SudokuSolver.h>
#include <test_helper.h>

namespace {

void
runTest(SolverType solverType, const std::string& path)
{
    std::unique_ptr<SudokuSolver> solver;
    try {
        solver = SudokuSolver::create(solverType);
    } catch (const std::runtime_error& e) {
        FAIL();
        return;
    }

    const auto challenges = readSudokuChallenges(path);

    for (const auto& challenge : challenges) {
        const auto result = solver->solve(challenge.grid);
        EXPECT_NE(result, nullptr);

        if (result) {
            int errCnt = 0;
            if (!challenge.isValidSolution(result.get(), &errCnt)) {
                std::cout << "Failed with " << errCnt << " errors." << std::endl;
                SudokuGrid::printGrid(std::cout, *result, false);

                std::cout << *result << '\n';
                std::cout << challenge.solution << '\n';
                std::cout << challenge.grid << '\n';
            }
            EXPECT_EQ(errCnt, 0);
        }
    }
}

TEST(solver, test_test_sudokus)
{
    runTest(SolverType::Dlx, "./solver/tests/test_sudokus.txt");
}

TEST(solver, test_any_sudokus)
{
    runTest(SolverType::Dlx, "./solver/tests/any.txt");
}

TEST(solver, test_simple_sudokus)
{
    runTest(SolverType::Dlx, "./solver/tests/simple.txt");
}

TEST(solver, test_easy_sudokus)
{
    runTest(SolverType::Dlx, "./solver/tests/easy.txt");
}

TEST(solver, test_intermediate_sudokus)
{
    runTest(SolverType::Dlx, "./solver/tests/intermediate.txt");
}

TEST(solver, test_expert_sudokus)
{
    runTest(SolverType::Dlx, "./solver/tests/expert.txt");
}

TEST(solver, test_test_sudokus_constraint)
{
    runTest(SolverType::Constraint, "./solver/tests/test_sudokus.txt");
}

TEST(solver, test_any_sudokus_constraint)
{
    runTest(SolverType::Constraint, "./solver/tests/any.txt");
}

TEST(solver, test_simple_sudokus_constraint)
{
    runTest(SolverType::Constraint, "./solver/tests/simple.txt");
}

TEST(solver, test_easy_sudokus_constraint)
{
    runTest(SolverType::Constraint, "./solver/tests/easy.txt");
}

TEST(solver, test_intermediate_sudokus_constraint)
{
    runTest(SolverType::Constraint, "./solver/tests/intermediate.txt");
}

TEST(solver, test_expert_sudokus_constraint)
{
    runTest(SolverType::Constraint, "./solver/tests/expert.txt");
}
}
