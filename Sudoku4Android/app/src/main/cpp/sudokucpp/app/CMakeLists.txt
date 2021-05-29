cmake_minimum_required(VERSION 3.18)

project(sudoku_app CXX)
set(CMAKE_CXX_STANDARD 17)

add_executable(sudoku_app src/app.cpp)

target_include_directories(sudoku_app PRIVATE
        sudoku_utils
        sudoku_detector
        digit_classifier
        include
        )
target_link_libraries(sudoku_app PRIVATE
        sudoku_utils
        sudoku_detector
        digit_classifier
        )

