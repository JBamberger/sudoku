//
// Created by jannik on 18.03.2022.
//

#ifndef SUDOKU4ANDROID_TEST_HELPER_H
#define SUDOKU4ANDROID_TEST_HELPER_H

#include "SudokuSolver.h"
#include <string>
#include <vector>

struct SudokuChallenge
{
    SudokuGrid grid;
    SudokuGrid solution;

    explicit SudokuChallenge(const std::string& line);
    bool isValidSolution(const SudokuGrid* result, int* errorCount) const;
};

std::vector<SudokuChallenge>
readSudokuChallenges(const std::string& path);

#endif // SUDOKU4ANDROID_TEST_HELPER_H
