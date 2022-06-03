add_library(digit_classifier STATIC
        src/CellClassifier.cpp include/CellClassifier.h)

target_include_directories(digit_classifier PUBLIC
        ${OpenCV_INCLUDE_DIRS}
        sudoku_utils
        include
        )
target_link_libraries(digit_classifier PUBLIC
        ${OpenCV_LIBS}
        sudoku_utils
        )

IF (BUILD_TESTING)
    add_executable(digit_classifier_test tests/test_classifier.cpp)
    target_include_directories(digit_classifier_test PUBLIC digit_classifier)
    target_link_libraries(digit_classifier_test PUBLIC digit_classifier)
ENDIF ()