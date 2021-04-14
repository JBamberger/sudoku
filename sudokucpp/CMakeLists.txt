cmake_minimum_required(VERSION 3.14)

#if (DEFINED ENV{VCPKG_ROOT} AND NOT DEFINED CMAKE_TOOLCHAIN_FILE)
#    set(CMAKE_TOOLCHAIN_FILE "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake" CACHE STRING "")
#endif ()

project(sudoku)
set(CMAKE_CXX_STANDARD 17)

find_package(Torch REQUIRED)
find_package(OpenCV 4 REQUIRED core imgcodecs imgproc highgui videoio)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(sudoku
        src/app.cpp
        #        src/DigitClassifier.cpp include/DigitClassifier.h
        #        src/SudokuDetector.cpp include/SudokuDetector.h
        #        src/SudokuSolver.cpp include/SudokuSolver.h
        #        src/SudokuDetector.cpp src/SudokuDetector.h
        )
target_include_directories(sudoku PRIVATE
        ${OpenCV_INCLUDE_DIRS}
        ${TORCH_INCLUDE_DIRS}
        include
        )
target_link_libraries(sudoku PRIVATE
        ${TORCH_LIBRARIES}
        ${OpenCV_LIBS}
        )

# The following code block is suggested to be used on Windows.
# According to https://github.com/pytorch/pytorch/issues/25457,
# the DLLs need to be copied to avoid memory errors.
if (MSVC)
    file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
    add_custom_command(TARGET sudoku
            POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${TORCH_DLLS}
            $<TARGET_FILE_DIR:sudoku>)
endif (MSVC)
