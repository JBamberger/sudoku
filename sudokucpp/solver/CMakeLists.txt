cmake_minimum_required(VERSION 3.14)

project(sudoku_solver)
set(CMAKE_CXX_STANDARD 17)

add_library(sudoku_solver
        src/SudokuSolver.cpp include/SudokuSolver.h
        )
target_include_directories(sudoku_solver PUBLIC
        include
        )
target_link_libraries(sudoku_solver PRIVATE)