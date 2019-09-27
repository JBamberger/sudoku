cmake_minimum_required(VERSION 3.14)

if(DEFINED ENV{VCPKG_ROOT} AND NOT DEFINED CMAKE_TOOLCHAIN_FILE)
    set(CMAKE_TOOLCHAIN_FILE "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake" CACHE STRING "")
endif()

project(sudoku)
set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV 3 REQUIRED core imgcodecs imgproc highgui videoio objdetect ml)

add_executable(sudoku 
    src/app.cpp
    src/DigitClassifier.cpp
	src/SudokuDetector.cpp
	src/SudokuSolver.cpp
)
target_include_directories(sudoku PRIVATE
	${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(sudoku PRIVATE
    ${OpenCV_LIBS}
)
