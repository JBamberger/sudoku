cmake_minimum_required(VERSION 3.14)

#if (DEFINED ENV{VCPKG_ROOT} AND NOT DEFINED CMAKE_TOOLCHAIN_FILE)
#    set(CMAKE_TOOLCHAIN_FILE "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake" CACHE STRING "")
#endif ()

project(sudoku_app CXX)
set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV 4 REQUIRED core imgcodecs imgproc highgui videoio ximgproc dnn)

add_subdirectory(util)
add_subdirectory(detector)
add_subdirectory(classifier)


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

