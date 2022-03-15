/*
 * Copyright (C) 2017 The Android Open Source Project
 * Modifications Copyright (C) 2022 Jannik Bamberger
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef __SUDOKU_APPLICATION_H__
#define __SUDOKU_APPLICATION_H__

#include <android/native_window.h>
#include <android_native_app_glue.h>
#include <functional>
#include <thread>

#include "camera_manager.h"

class SudokuApplication {
public:
    explicit SudokuApplication(android_app *app);

    ~SudokuApplication();

    // Application events
    void onAppInitWindow();

    void onAppConfigChange();

    void onAppTermWindow();

    void pollAndDrawFrame();

    void onCameraPermission(bool granted);

    void onTakePhoto();

    // Window surface parameters
    void saveBufferGeometry(ANativeWindow *window);

    void restoreBufferGeometry(ANativeWindow *window) const;

    // Camera and ImageReader setup and teardown
    void createCamera();

    void deleteCamera();

private:
    // Calls back into the java application
    void jniRequestCameraPermission();

    void jniUpdateUI();

    void jniOnPhotoTaken(const char *fileName);

    int jniGetDisplayRotation();


    struct android_app *app_;

    ImageFormat savedNativeWinRes_;
    bool cameraGranted_;
    int rotation_;
    volatile bool cameraReady_;

    NDKCamera *camera_;
    ImageReader *yuvReader_;
    ImageReader *jpgReader_;

    int32_t computeImageRotation();
};

SudokuApplication *getApplication();

#endif  // __SUDOKU_APPLICATION_H__
