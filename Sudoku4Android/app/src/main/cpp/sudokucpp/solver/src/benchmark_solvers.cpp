//
// Created by jannik on 10.08.2021.
//

#include <SudokuSolver.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <ctime>
#include <ratio>
#include <chrono>

namespace fs = std::filesystem;

std::vector<std::pair<SudokuGrid, SudokuGrid>>
readSudokus(const std::string& file)
{
    std::string line;
    std::ifstream sudokuListFile(file);

    if (!sudokuListFile.is_open()) {
        std::cerr << "Could not open sudoku list." << std::endl;
        exit(EXIT_FAILURE);
    }

    std::vector<std::pair<SudokuGrid, SudokuGrid>> sudokus;
    while (std::getline(sudokuListFile, line)) {
        SudokuGrid grid{};
        for (int i = 0; i < 81; i++) {
            char cellValue = line.at(i);
            grid[i] = cellValue == '.' ? 0 : cellValue - '0';
        }

        SudokuGrid solution{};
        for (int i = 0; i < 81; i++) {
            char cellValue = line.at(82 + i); // offset by 81 for sudoku and 1 for ';' divider
            solution[i] = cellValue - '0';
        }
        sudokus.emplace_back(grid, solution);
    }

    return sudokus;
}

void
evalConfig(SolverType type, const std::string& file)
{
    using namespace std::chrono;

    auto sudokus = readSudokus(file);
    auto solver = SudokuSolver::create(SolverType::Dlx);

    std::vector<double> times;

    int countCorrect = 0;
    for (auto [grid, solution] : sudokus) {
        auto start = high_resolution_clock::now();

        auto result = solver->solve(grid);

        auto end = high_resolution_clock::now();
        duration<double, std::milli> dur = end - start;
        times.push_back(dur.count());

        if (result == nullptr) {
            std::cerr << "Failed to solve sudoku!" << std::endl;
        } else {
            int errCnt = 0;
            for (int i = 0; i < 81; i++) {
                if (result->at(i) != solution[i]) {
                    errCnt++;
                }
            }
            if (errCnt > 0) {
                std::cerr << "Failed with " << errCnt << " errors." << std::endl;
                printGrid(*result, false);

                printGrid(*result);
                printGrid(solution);
                printGrid(grid);
            } else {
                countCorrect++;
//                std::cout << "Solved correctly" << std::endl;
            }
        }
    }

    double totalTime = 0.0;
    for (auto t : times) {
        totalTime += t;
    }

    double mean = totalTime / static_cast<double>(times.size());

    double var = 0.0;
    for (auto t : times) {
        double x = t - mean;
        var += x * x;
    }
    double stddev = sqrt(var);


    std::cout << "Solved " << countCorrect << "/" << sudokus.size() << " Sudokus correct. Took "
              << totalTime << "ms. (Mean: " << mean << "ms StdDev: " << stddev << "ms)" << std::endl;
}

int
main()
{
    evalConfig(SolverType::Dlx, "./solver/tests/test_sudokus.txt");
    evalConfig(SolverType::Dlx, "./solver/tests/any.txt");
    evalConfig(SolverType::Dlx, "./solver/tests/simple.txt");
    evalConfig(SolverType::Dlx, "./solver/tests/easy.txt");
    evalConfig(SolverType::Dlx, "./solver/tests/intermediate.txt");
    evalConfig(SolverType::Dlx, "./solver/tests/expert.txt");

    evalConfig(SolverType::Constraint, "./solver/tests/test_sudokus.txt");
    evalConfig(SolverType::Constraint, "./solver/tests/any.txt");
    evalConfig(SolverType::Constraint, "./solver/tests/simple.txt");
    evalConfig(SolverType::Constraint, "./solver/tests/easy.txt");
    evalConfig(SolverType::Constraint, "./solver/tests/intermediate.txt");
    evalConfig(SolverType::Constraint, "./solver/tests/expert.txt");
}