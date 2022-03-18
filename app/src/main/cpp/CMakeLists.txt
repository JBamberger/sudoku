cmake_minimum_required(VERSION 3.18)
project(sudokucpp)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV 4.5 REQUIRED core imgcodecs imgproc highgui videoio dnn)
find_library(log-lib log)

add_subdirectory(sudokucpp)

IF (BUILD_DESKTOP)
    add_subdirectory(desktop_app)
ENDIF ()

IF (BUILD_ANDROID)
    add_subdirectory(android_app)
    add_subdirectory(jni_binding)
ENDIF ()