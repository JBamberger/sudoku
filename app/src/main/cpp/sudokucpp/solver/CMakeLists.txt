add_library(sudoku_solver
        src/dlx.cpp
        src/DlxSudokuSolver.cpp src/DlxSudokuSolver.h
        src/ConstraintSudokuSolver.cpp src/ConstraintSudokuSolver.h
        src/SudokuSolver.cpp include/SudokuSolver.h
        )
target_link_libraries(sudoku_solver PUBLIC
        sudoku_utils)
target_include_directories(sudoku_solver PUBLIC
        sudoku_utils
        include)


if (PACKAGE_TESTS)
    add_subdirectory(tests)
endif ()