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

#include "sudoku_application.h"
#include "native_debug.h"


static std::unique_ptr<SudokuApplication> application;

SudokuApplication *getApplication() {
    ASSERT(application, "Application is not initialized!");
    return application.get();
}


static void handleAndroidCommand(struct android_app *app, int32_t cmd) {
    auto *engine = reinterpret_cast<SudokuApplication *>(app->userData);
    switch (cmd) {
        case APP_CMD_INIT_WINDOW:
            if (app->window != nullptr) {
                engine->saveBufferGeometry(app->window);
                engine->onAppInitWindow();
            }
            break;
        case APP_CMD_TERM_WINDOW:
            engine->onAppTermWindow();
            engine->restoreBufferGeometry(app->window);
            break;
        case APP_CMD_CONFIG_CHANGED:
            engine->onAppConfigChange();
            break;
    }
}

/**
 * Application main loop: Handles incoming system events, for example the permission request result.
 *
 * Polls the camera for new frames and dispatches the image processing tasks.
 */
extern "C" void android_main(struct android_app *state) {
    application = std::make_unique<SudokuApplication>(state);

    state->userData = reinterpret_cast<void *>(application.get());
    state->onAppCmd = handleAndroidCommand;

    while (true) {
        int events;
        struct android_poll_source *source;
        while (ALooper_pollAll(0, nullptr, &events, (void **) &source) >= 0) {
            if (source != nullptr) {
                source->process(state, source);
            }

            if (state->destroyRequested != 0) {
                LOGI("CameraEngine thread destroy requested!");
                application->deleteCamera();
                application = nullptr;
                return;
            }
        }

        application->pollAndDrawFrame();
    }
}

void SudokuApplication::onAppInitWindow() {
    if (!cameraGranted_) {
        // Not permitted to use camera yet, ask(again) and defer other events
        jniRequestCameraPermission();
        return;
    }

    rotation_ = jniGetDisplayRotation();

    createCamera();
    ASSERT(camera_, "CameraCreation Failed");

    jniUpdateUI();

    // NativeActivity end is ready to display, start pulling images
    cameraReady_ = true;
    camera_->StartPreview(true);
}

void SudokuApplication::onAppTermWindow() {
    cameraReady_ = false;
    deleteCamera();
}

void SudokuApplication::onAppConfigChange() {
    int newRotation = jniGetDisplayRotation();

    if (newRotation != rotation_) {
        onAppTermWindow();
        rotation_ = newRotation;
        onAppInitWindow();
    }
}

void SudokuApplication::saveBufferGeometry(ANativeWindow *window) {
    savedNativeWinRes_.width = ANativeWindow_getWidth(window);
    savedNativeWinRes_.height = ANativeWindow_getHeight(window);
    savedNativeWinRes_.format = ANativeWindow_getFormat(window);
}

void SudokuApplication::restoreBufferGeometry(ANativeWindow *window) const {
    ANativeWindow_setBuffersGeometry(
            window,
            savedNativeWinRes_.width,
            savedNativeWinRes_.height,
            savedNativeWinRes_.format);
}
