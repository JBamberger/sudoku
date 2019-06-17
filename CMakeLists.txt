project(sudoku)
cmake_minimum_required(VERSION 2.8.12)
set(sudoku_VERSION_MAJOR 0)
set(sudoku_VERSION_MINOR 1)
set(sudoku_VERSION_PATCH 0)
set(sudoku_VERSION ${sudoku_VERSION_MAJOR}.${sudoku_VERSION_MINOR}.${sudoku_VERSION_PATCH})

# Check if a supported compiler is used and add c++14 flag:
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 4.9)
        message(FATAL_ERROR "Need at least gcc 4.9 to compile.")
    endif()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -pthread")
elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
    if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 19)
        message(FATAL_ERROR "Visual Studio 2015 or newer is required.")
    endif()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /bigobj")
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 3.6)
        message(FATAL_ERROR "Clang below version 3.6 will most likely not work. Please upgrade your compiler.")
    endif()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -pthreads")
else() # no GNU, no MSVC, no Clang
    message(WARNING "You are using an unsupported compiler. Compilation has only been tested with MSVC, GCC and Clang.")
    include(CheckCXXCompilerFlag)
    check_cxx_compiler_flag(-std=c++14 HAS_CXX14_FLAG)
    if(HAS_CXX14_FLAG)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")
    else()
        message(FATAL_ERROR "Your compiler doesn't support the '-std=c++14' flag.")
    endif()
endif()

# Find dependencies:
find_package(OpenCV 3 REQUIRED core imgcodecs imgproc highgui videoio objdetect ml)
message(STATUS "OpenCV include dir found at ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV library dir found at ${OpenCV_LIB_DIR}")
set_target_properties(${OpenCV_LIBS} PROPERTIES MAP_IMPORTED_CONFIG_RELWITHDEBINFO RELEASE)


# Add header includes:
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(sudoku app.cpp app.h DigitClassifier.cpp DigitClassifier.h SudokuDetector.cpp SudokuDetector.h)
target_link_libraries(sudoku ${OpenCV_LIBS})

# install targets:
install(TARGETS sudoku DESTINATION bin)
