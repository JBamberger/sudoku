#add_test(NAME testname COMMAND $<TARGET_FILE:${TESTNAME}>)
#
#add_executable(sudoku_solver_test
#        tests/solverTest.cpp
#        )
#target_include_directories(sudoku_solver_test PRIVATE sudoku_solver)
#target_link_libraries(sudoku_solver_test PRIVATE sudoku_solver)


# Instructions from VCPKG for GTest usage:
#enable_testing()
#find_package(GTest CONFIG REQUIRED)
#target_link_libraries(main PRIVATE GTest::gtest GTest::gtest_main GTest::gmock GTest::gmock_main)
#add_test(AllTestsInMain main)


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
