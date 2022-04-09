
find_package(fmt REQUIRED)

add_executable(benchmark_sudoku_detection detection_benchmark.cpp)
target_link_libraries(benchmark_sudoku_detection PRIVATE sudoku_detector fmt::fmt)
target_include_directories(benchmark_sudoku_detection PRIVATE include)

#package_add_test_with_libraries(test_sudoku_grid test_sudoku_grid.cpp sudoku_utils "${PROJECT_SOURCE_DIR}")
