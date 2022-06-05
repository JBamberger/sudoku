add_library(sudoku_detector STATIC
        src/SudokuDetector.cpp include/SudokuDetector.h
        src/SudokuDetection.cpp include/SudokuDetection.h
        src/detect_sudoku.cpp src/detect_sudoku.h
        )
target_include_directories(sudoku_detector PUBLIC
        ${OpenCV_INCLUDE_DIRS}
        sudoku_utils
        digit_classifier
        sudoku_solver
        include
        )
target_link_libraries(sudoku_detector PUBLIC
        ${OpenCV_LIBS}
        sudoku_utils
        digit_classifier
        sudoku_solver
        )

if (BUILD_TESTING)
    add_executable(benchmark_sudoku_detection
            tests/detection_benchmark.cpp
            src/detect_sudoku.h
            )
    target_link_libraries(benchmark_sudoku_detection PRIVATE sudoku_detector)
    target_include_directories(benchmark_sudoku_detection PRIVATE include src)
    #target_link_options(benchmark_sudoku_detection PRIVATE /PROFILE)

    #package_add_test_with_libraries(test_sudoku_grid test_sudoku_grid.cpp sudoku_utils "${PROJECT_SOURCE_DIR}")
endif ()