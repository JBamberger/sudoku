cmake_minimum_required(VERSION 3.18)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV 4.5 REQUIRED java core imgcodecs imgproc highgui videoio dnn)
find_library(log-lib log)

add_subdirectory(sudokucpp)
add_subdirectory(desktop_app)
add_subdirectory(android_app)

add_subdirectory(jni_binding)
