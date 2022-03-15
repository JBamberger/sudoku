cmake_minimum_required(VERSION 3.18)

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
