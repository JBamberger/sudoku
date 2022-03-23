#include <cassert>
#include <fstream>
#include <iostream>

#include "test_helper.h"

SudokuChallenge::SudokuChallenge(const std::string& line)
  : grid(9)
  , solution(9)
{
    assert(line.size() == 2 * grid.size() + 1);

    for (int i = 0; i < grid.size(); i++) {
        char cellValue = line.at(i);
        grid[i] = cellValue == '.' ? 0 : cellValue - '0';
    }

    for (int i = 0; i < grid.size(); i++) {
        char cellValue = line.at(grid.size() + 1 + i); // offset by 81 for sudoku and 1 for ';' divider
        solution[i] = cellValue - '0';
    }
}

bool
SudokuChallenge::isValidSolution(const SudokuGrid* result, int* errorCount) const
{
    if (!result || result->size() != solution.size()) {
        *errorCount = -1;
        return false;
    }

    *errorCount = 0;
    for (int i = 0; i < grid.size(); i++) {
        if (result->at(i) != solution[i]) {
            (*errorCount)++;
        }
    }
    return (*errorCount) == 0;
}

std::vector<SudokuChallenge>
readSudokuChallenges(const std::string& path)
{
    std::string line;
    std::ifstream sudokuListFile;
    sudokuListFile.open(path);

    if (!sudokuListFile.is_open()) {
        std::cerr << "Could not open sudoku list." << std::endl;
        exit(EXIT_FAILURE);
    }

    std::vector<SudokuChallenge> challenges;
    while (std::getline(sudokuListFile, line)) {
        challenges.emplace_back(line);
    }

    return challenges;
}