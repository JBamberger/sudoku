#include "gtest/gtest.h"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

#include <SudokuSolver.h>

namespace fs = std::filesystem;

namespace {
TEST(solver, test1)
{
    std::cout << fs::absolute(".") << std::endl;
    std::string line;
    std::ifstream sudokuListFile;
    sudokuListFile.open("./solver/tests/test_sudokus.txt");

    if (!sudokuListFile.is_open()) {
        std::cerr << "Could not open sudoku list." << std::endl;
        exit(EXIT_FAILURE);
    }

    auto solver = SudokuSolver::create(SolverType::Dlx);

    while (std::getline(sudokuListFile, line)) {
        //        std::cout << line << std::endl;

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

        auto result = solver->solve(grid);
        //        std::cout << (result == nullptr ? "Not Solved" : "solved") << std::endl;

        EXPECT_NE(result, nullptr);

        if (result != nullptr) {
            int errCnt = 0;
            for (int i = 0; i < 81; i++) {
                if (result->at(i) != solution[i]) {
                    errCnt++;
                }
            }
            if (errCnt > 0) {
                std::cout << "Failed with " << errCnt << " errors." << std::endl;
                printGrid(*result, false);

                printGrid(*result);
                printGrid(solution);
                printGrid(grid);
            } else {
                std::cout << "Solved correctly" << std::endl;
            }
            EXPECT_EQ(errCnt, 0);
        } else {
            std::cout << "Not solved!" << std::endl;
        }
    }

    // 4 . . | . . . | 8 . 5
    // . 3 . | . . . | . . .
    // . . . | 7 . . | . . .
    // ------+-------+------
    // . 2 . | . . . | . 6 .
    // . . . | . 8 . | 4 . .
    // . . . | . 1 . | . . .
    // ------+-------+------
    // . . . | 6 . 3 | . 7 .
    // 5 . . | 2 . . | . . .
    // 1 . 4 | . . . | . . .
    //

    // 4 1 7 | 3 6 9 | 8 2 5
    // 6 3 2 | 1 5 8 | 9 4 7
    // 9 5 8 | 7 2 4 | 3 1 6
    // ------+-------+------
    // 8 2 5 | 4 3 7 | 1 6 9
    // 7 9 1 | 5 8 6 | 4 3 2
    // 3 4 6 | 9 1 2 | 7 5 8
    // ------+-------+------
    // 2 8 9 | 6 4 3 | 5 7 1
    // 5 7 3 | 2 9 1 | 6 8 4
    // 1 6 4 | 8 7 5 | 2 9 3
}
}
