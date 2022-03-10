# For more information about using CMake with Android Studio, read the
# documentation: https://d.android.com/studio/projects/add-native-code.html

# Sets the minimum version of CMake required to build the native library.

cmake_minimum_required(VERSION 3.18)

project(sudoku4android CXX)
set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV 4.5 REQUIRED java core imgcodecs imgproc highgui videoio dnn)


add_subdirectory(sudokucpp)
find_library(log-lib log)

add_library(mixed_sample SHARED
        jni_part.cpp
        )
target_link_libraries(mixed_sample PUBLIC
        ${OpenCV_LIBRARIES}
        sudoku_utils
        sudoku_detector
        digit_classifier
        )
target_include_directories(mixed_sample PUBLIC
        sudoku_utils
        sudoku_detector
        digit_classifier
        includes
        )
