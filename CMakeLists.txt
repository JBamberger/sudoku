cmake_minimum_required(VERSION 3.14)

if (DEFINED ENV{VCPKG_ROOT} AND NOT DEFINED CMAKE_TOOLCHAIN_FILE)
    set(CMAKE_TOOLCHAIN_FILE "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake" CACHE STRING "")
endif ()

project(sudoku)
set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV 4 REQUIRED core imgcodecs imgproc highgui videoio objdetect ml)

add_executable(sudoku
        src/app.cpp include/app.h
        src/DigitClassifier.cpp include/DigitClassifier.h
        src/SudokuDetector.cpp include/SudokuDetector.h
        src/SudokuSolver.cpp include/SudokuSolver.h
        )
target_include_directories(sudoku PRIVATE
        ${OpenCV_INCLUDE_DIRS}
        include
        )
target_link_libraries(sudoku PRIVATE
        ${OpenCV_LIBS}
        )
