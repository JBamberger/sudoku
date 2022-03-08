cmake_minimum_required(VERSION 3.16)

#if (DEFINED ENV{VCPKG_ROOT} AND NOT DEFINED CMAKE_TOOLCHAIN_FILE)
#    set(CMAKE_TOOLCHAIN_FILE "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake" CACHE STRING "")
#endif ()

project(sudoku_app CXX)
set(CMAKE_CXX_STANDARD 17)

if (CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    include(CTest)
endif ()

option(PACKAGE_TESTS "Build the tests" ON)

message(STATUS "PACKAGE_TESTS=${PACKAGE_TESTS}")

if (PACKAGE_TESTS)
    enable_testing()
    include(GoogleTest)
    add_subdirectory(solver/tests)
endif ()

find_package(OpenCV 4 REQUIRED core imgcodecs imgproc highgui videoio dnn)

add_subdirectory(util)
add_subdirectory(detector)
add_subdirectory(classifier)
add_subdirectory(solver)
add_subdirectory(app)
add_subdirectory(detection_benchmark)
