add_executable(benchmark_sudoku_detection detection_benchmark.cpp)

target_include_directories(benchmark_sudoku_detection PRIVATE
        sudoku_utils
        sudoku_detector
        digit_classifier
        include
        )
target_link_libraries(benchmark_sudoku_detection PRIVATE
        sudoku_utils
        sudoku_detector
        digit_classifier
        )

