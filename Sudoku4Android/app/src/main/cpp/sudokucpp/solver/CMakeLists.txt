#cmake_minimum_required(VERSION 3.14)

#project(sudoku_solver)
#set(CMAKE_CXX_STANDARD 17)

add_library(sudoku_solver
        src/dlx.cpp src/dlx.h
        src/SudokuSolver.cpp include/SudokuSolver.h
        )
target_include_directories(sudoku_solver PUBLIC
        include
        )
#target_link_libraries(sudoku_solver PUBLIC)

#add_test(NAME testname COMMAND $<TARGET_FILE:${TESTNAME}>)
#
#add_executable(sudoku_solver_test
#        tests/solverTest.cpp
#        )
#target_include_directories(sudoku_solver_test PRIVATE sudoku_solver)
#target_link_libraries(sudoku_solver_test PRIVATE sudoku_solver)


add_executable(dlx_test
        src/dlx.cpp src/dlx.h
        src/SudokuSolver.cpp include/SudokuSolver.h
        )
target_include_directories(dlx_test PUBLIC
        include
        )