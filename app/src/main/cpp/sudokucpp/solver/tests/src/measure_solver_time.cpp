
#include <chrono>
#include <fmt/format.h>
#include <string>
#include <array>

#include "test_helper.h"
#include "SudokuSolver.h"

void
measure(SolverType solverType, const std::string& path)
{
    using namespace std::chrono;
    std::unique_ptr<SudokuSolver> solver = SudokuSolver::create(solverType);

    const auto challenges = readSudokuChallenges(path);

    std::vector<double> times;
    for (const auto& challenge : challenges) {
        auto start = high_resolution_clock::now();

        const auto result = solver->solve(challenge.grid);

        auto end = high_resolution_clock::now();
        duration<double, std::milli> dur = end - start;
        times.push_back(dur.count());
    }

    double totalTime = 0.0;
    for (auto t : times) {
        totalTime += t;
    }

    double mean = totalTime / static_cast<double>(times.size());

    double min = std::numeric_limits<double>::max();
    double max = std::numeric_limits<double>::min();
    double var = 0.0;
    for (auto t : times) {
        double x = t - mean;
        var += x * x;

        if (t < min) {
            min = t;
        }
        if (t > max) {
            max = t;
        }
    }
    double stddev = sqrt(var);

    fmt::print("{:9.2f} {:9.2f} {:9.2f} {:9.2f} {:9.2f} path={}\n", totalTime, mean, stddev, min, max, path);
}

int
main()
{
    std::array<std::string, 6> files{
                "./solver/tests/test_sudokus.txt",
        "./solver/tests/any.txt",
                "./solver/tests/simple.txt",
                "./solver/tests/easy.txt",         "./solver/tests/intermediate.txt", "./solver/tests/expert.txt",
    };

    fmt::print("DLX:\n");
    fmt::print("totalTime      mean    stddev       min       max path\n");
    for (const auto& file : files) {
        measure(SolverType::Dlx, file);
    }
    //    fmt::print("Constraint:\n");
    //    fmt::print("totalTime      mean    stddev       min       max path\n");
    //    for (const auto& file : files) {
    //        measure(SolverType::Constraint, file);
    //    }

    //    measure(SolverType::Constraint, "./solver/tests/test_sudokus.txt");
    //    measure(SolverType::Constraint, "./solver/tests/any.txt");
    //    measure(SolverType::Constraint, "./solver/tests/simple.txt");
    //    measure(SolverType::Constraint, "./solver/tests/easy.txt");
    //    measure(SolverType::Constraint, "./solver/tests/intermediate.txt");
    //    measure(SolverType::Constraint, "./solver/tests/expert.txt");
    //
    //    measure(SolverType::Dlx, "./solver/tests/test_sudokus.txt");
    //    measure(SolverType::Dlx, "./solver/tests/any.txt");
    //    measure(SolverType::Dlx, "./solver/tests/simple.txt");
    //    measure(SolverType::Dlx, "./solver/tests/easy.txt");
    //    measure(SolverType::Dlx, "./solver/tests/intermediate.txt");
    //    measure(SolverType::Dlx, "./solver/tests/expert.txt");
}
