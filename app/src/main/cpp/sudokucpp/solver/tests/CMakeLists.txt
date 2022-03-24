
find_package(fmt REQUIRED)
find_package(benchmark CONFIG REQUIRED)

add_library(test_helper src/test_helper.cpp)
target_include_directories(test_helper PUBLIC sudoku_solver dlx include)
target_link_libraries(test_helper PUBLIC sudoku_solver)

package_add_test_with_libraries(test-sudoku-solver src/solverTest.cpp test_helper "${PROJECT_SOURCE_DIR}")

add_executable(solver_mat_init_benchmark src/mat_init_benchmark.cpp)
target_link_libraries(solver_mat_init_benchmark PRIVATE
        benchmark::benchmark
        benchmark::benchmark_main
        test_helper)

add_executable(solver_benchmark src/measure_solver_time.cpp)
target_link_libraries(solver_benchmark test_helper fmt::fmt)
target_include_directories(solver_benchmark PRIVATE test_helper include)
target_link_options(solver_benchmark PRIVATE /PROFILE)
