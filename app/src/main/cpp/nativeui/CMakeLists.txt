#
# Copyright (C) 2022 Jannik Bamberger
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

cmake_minimum_required(VERSION 3.18)

add_library(native_app_glue STATIC
        ${ANDROID_NDK}/sources/android/native_app_glue/android_native_app_glue.c)
target_include_directories(native_app_glue PUBLIC
        ${ANDROID_NDK}/sources/android/native_app_glue)

# Export ANativeActivity_onCreate() (see https://github.com/android-ndk/ndk/issues/381)
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -u ANativeActivity_onCreate")

add_library(nativeui SHARED
        src/android_main.cpp
        src/sudoku_application.cpp
        src/sudoku_application_jni.cpp
        src/cv_image_reader.cpp
        src/camera_manager.cpp
        src/camera_listeners.cpp
        src/camera_utils.cpp)
target_link_libraries(nativeui
        android
        log
        m
        native_app_glue
        camera2ndk
        mediandk
        ${OpenCV_LIBRARIES}
        sudoku_utils
        sudoku_detector
        digit_classifier
)
target_include_directories(nativeui PRIVATE
        include
        sudoku_utils
        sudoku_detector
        digit_classifier
)
