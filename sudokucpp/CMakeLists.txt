cmake_minimum_required(VERSION 3.14)

#if (DEFINED ENV{VCPKG_ROOT} AND NOT DEFINED CMAKE_TOOLCHAIN_FILE)
#    set(CMAKE_TOOLCHAIN_FILE "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake" CACHE STRING "")
#endif ()

project(sudoku_app CXX)
set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV 4 REQUIRED core imgcodecs imgproc highgui videoio ximgproc dnn)

add_executable(sudoku_app
        src/app.cpp
        src/SudokuDetector.cpp include/SudokuDetector.h
        src/utils.cpp include/utils.h
        include/config.h
        )
target_include_directories(sudoku_app PRIVATE
        ${OpenCV_INCLUDE_DIRS}
        include
        )
target_link_libraries(sudoku_app PRIVATE
        ${OpenCV_LIBS}
        )

add_subdirectory(classifier)