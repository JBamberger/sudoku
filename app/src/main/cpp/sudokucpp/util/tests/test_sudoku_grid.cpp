#include "gtest/gtest.h"
#include "sudoku_grid.h"
#include <sstream>

TEST(sudoku_grid, test_sudoku_grid_init_min_size)
{
    SudokuGrid grid1(1);
    EXPECT_EQ(grid1.sideLen, 1);
    EXPECT_EQ(grid1.boxLen, 1);
    EXPECT_EQ(grid1.size(), 1);

    SudokuGrid grid4(4);
    EXPECT_EQ(grid4.sideLen, 4);
    EXPECT_EQ(grid4.boxLen, 2);
    EXPECT_EQ(grid4.size(), 16);

    SudokuGrid grid9(9);
    EXPECT_EQ(grid9.sideLen, 9);
    EXPECT_EQ(grid9.boxLen, 3);
    EXPECT_EQ(grid9.size(), 81);

    SudokuGrid grid16(16);
    EXPECT_EQ(grid16.sideLen, 16);
    EXPECT_EQ(grid16.boxLen, 4);
    EXPECT_EQ(grid16.size(), 256);

    SudokuGrid grid1764(1764);
    EXPECT_EQ(grid1764.sideLen, 1764);
    EXPECT_EQ(grid1764.boxLen, 42);
    EXPECT_EQ(grid1764.size(), 3111696);
}

TEST(sudoku_grid, test_sudoku_grid_printGridFlat4x4empty)
{
    SudokuGrid grid(4);

    std::stringstream out;
    SudokuGrid::printGrid(out, grid, true);
    EXPECT_EQ(out.str(), "...."
                         "...."
                         "...."
                         "....");
}

TEST(sudoku_grid, test_sudoku_grid_printGridFlat9x9empty)
{
    SudokuGrid grid(9);

    std::stringstream out;
    SudokuGrid::printGrid(out, grid, true);
    EXPECT_EQ(out.str(), "........."
                         "........."
                         "........."
                         "........."
                         "........."
                         "........."
                         "........."
                         "........."
                         ".........");
}

TEST(sudoku_grid, test_sudoku_grid_printGrid4x4filled)
{
    SudokuGrid grid(4);
    grid.at(0, 0) = 1;
    grid.at(0, 1) = 2;
    grid.at(0, 2) = 3;
    grid.at(0, 3) = 4;

    grid.at(3, 0) = 3;
    grid.at(3, 1) = 4;
    grid.at(3, 2) = 3;
    grid.at(3, 3) = 4;

    grid.at(1, 0) = 2;
    grid.at(2, 0) = 2;

    grid.at(1, 3) = 1;
    grid.at(2, 3) = 1;

    std::stringstream out;
    SudokuGrid::printGrid(out, grid, true);
    EXPECT_EQ(out.str(), "1234"
                         "2..1"
                         "2..1"
                         "3434");
    std::stringstream outMl;
    SudokuGrid::printGrid(outMl, grid, false);
    EXPECT_EQ(outMl.str(), "12|34\n"
                         "2.|.1\n"
                         "--+--\n"
                         "2.|.1\n"
                         "34|34\n");
}

TEST(sudoku_grid, test_sudoku_grid_printGridFlat9x9filled)
{
    SudokuGrid grid(9);
    grid.at(0, 0) = 1;
    grid.at(0, 1) = 2;
    grid.at(0, 2) = 3;
    grid.at(0, 3) = 4;
    grid.at(0, 4) = 5;
    grid.at(0, 5) = 6;
    grid.at(0, 6) = 7;
    grid.at(0, 7) = 8;
    grid.at(0, 8) = 9;

    grid.at(8, 0) = 8;
    grid.at(8, 1) = 7;
    grid.at(8, 2) = 8;
    grid.at(8, 3) = 7;
    grid.at(8, 4) = 8;
    grid.at(8, 5) = 7;
    grid.at(8, 6) = 8;
    grid.at(8, 7) = 7;
    grid.at(8, 8) = 9;

    grid.at(1, 0) = 9;
    grid.at(2, 0) = 9;
    grid.at(3, 0) = 9;
    grid.at(4, 0) = 9;
    grid.at(5, 0) = 9;
    grid.at(6, 0) = 9;
    grid.at(7, 0) = 9;

    grid.at(1, 8) = 5;
    grid.at(2, 8) = 5;
    grid.at(3, 8) = 5;
    grid.at(4, 8) = 5;
    grid.at(5, 8) = 5;
    grid.at(6, 8) = 5;
    grid.at(7, 8) = 5;

    std::stringstream out;
    SudokuGrid::printGrid(out, grid, true);
    EXPECT_EQ(out.str(), "123456789"
                         "9.......5"
                         "9.......5"
                         "9.......5"
                         "9.......5"
                         "9.......5"
                         "9.......5"
                         "9.......5"
                         "878787879");
    std::stringstream outMl;
    SudokuGrid::printGrid(outMl, grid, false);
    EXPECT_EQ(outMl.str(), "123|456|789\n"
                           "9..|...|..5\n"
                           "9..|...|..5\n"
                           "---+---+---\n"
                           "9..|...|..5\n"
                           "9..|...|..5\n"
                           "9..|...|..5\n"
                           "---+---+---\n"
                           "9..|...|..5\n"
                           "9..|...|..5\n"
                           "878|787|879\n");
}

TEST(sudoku_grid, test_sudoku_grid_printGridFlat16x16)
{
    SudokuGrid grid(16);

    std::stringstream out;
    SudokuGrid::printGrid(out, grid, true);
    EXPECT_EQ(out.str(), "................"
                         "................"
                         "................"
                         "................"
                         "................"
                         "................"
                         "................"
                         "................"
                         "................"
                         "................"
                         "................"
                         "................"
                         "................"
                         "................"
                         "................"
                         "................");
    std::stringstream outMl;
    SudokuGrid::printGrid(outMl, grid, false);
    EXPECT_EQ(outMl.str(), "....|....|....|....\n"
                           "....|....|....|....\n"
                           "....|....|....|....\n"
                           "....|....|....|....\n"
                           "----+----+----+----\n"
                           "....|....|....|....\n"
                           "....|....|....|....\n"
                           "....|....|....|....\n"
                           "....|....|....|....\n"
                           "----+----+----+----\n"
                           "....|....|....|....\n"
                           "....|....|....|....\n"
                           "....|....|....|....\n"
                           "....|....|....|....\n"
                           "----+----+----+----\n"
                           "....|....|....|....\n"
                           "....|....|....|....\n"
                           "....|....|....|....\n"
                           "....|....|....|....\n");
}