cmake_minimum_required(VERSION 3.4.1)

add_library(native_app_glue STATIC
        ${ANDROID_NDK}/sources/android/native_app_glue/android_native_app_glue.c)
target_include_directories(native_app_glue PUBLIC
        ${ANDROID_NDK}/sources/android/native_app_glue)
#set_target_properties(native_app_glue PROPERTIES LINKER_LANGUAGE CXX)

#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Werror")
# Export ANativeActivity_onCreate(),
# Refer to: https://github.com/android-ndk/ndk/issues/381.
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -u ANativeActivity_onCreate")

add_library(nativeui SHARED
        src/android_main.cpp
        src/sudoku_application.cpp
        src/sudoku_application_jni.cpp
        src/image_reader.cpp
        src/camera_manager.cpp
        src/camera_listeners.cpp
        src/camera_utils.cpp)
target_link_libraries(nativeui
        android
        log
        m
        native_app_glue
        camera2ndk
        mediandk)
target_include_directories(nativeui PRIVATE include)
