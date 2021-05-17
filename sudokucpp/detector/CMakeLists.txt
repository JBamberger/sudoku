find_package(OpenCV 4 REQUIRED core imgcodecs imgproc highgui videoio ximgproc dnn)

add_library(sudoku_detector STATIC
        src/SudokuDetector.cpp include/SudokuDetector.h
        )
target_include_directories(sudoku_detector PUBLIC
        ${OpenCV_INCLUDE_DIRS}
        sudoku_utils
        digit_classifier
        include
        )
target_link_libraries(sudoku_detector PUBLIC
        ${OpenCV_LIBS}
        sudoku_utils
        digit_classifier
        )