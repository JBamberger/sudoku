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
#include <native_debug.h>

std::string jstring2string(JNIEnv *env, jstring jStr) {
    if (!jStr)
        return "";

    const jclass stringClass = env->GetObjectClass(jStr);
    const jmethodID getBytes = env->GetMethodID(stringClass, "getBytes", "(Ljava/lang/String;)[B");
    const jbyteArray stringJbytes = (jbyteArray) env->CallObjectMethod(jStr, getBytes,
                                                                       env->NewStringUTF("UTF-8"));

    size_t length = (size_t) env->GetArrayLength(stringJbytes);
    jbyte *pBytes = env->GetByteArrayElements(stringJbytes, NULL);

    std::string ret = std::string((char *) pBytes, length);
    env->ReleaseByteArrayElements(stringJbytes, pBytes, JNI_ABORT);

    env->DeleteLocalRef(stringJbytes);
    env->DeleteLocalRef(stringClass);
    return ret;
}


void SudokuApplication::jniRequestCameraPermission() {
    if (!app_) return;

    JNIEnv *env;
    ANativeActivity *activity = app_->activity;
    activity->vm->GetEnv((void **) &env, JNI_VERSION_1_6);

    activity->vm->AttachCurrentThread(&env, nullptr);

    jobject activityObj = env->NewGlobalRef(activity->clazz);
    jclass clz = env->GetObjectClass(activityObj);
    env->CallVoidMethod(activityObj, env->GetMethodID(clz, "requestCamera", "()V"));
    env->DeleteGlobalRef(activityObj);

    activity->vm->DetachCurrentThread();
}

int SudokuApplication::jniGetDisplayRotation() {
    ASSERT(app_, "Application is not initialized");

    JNIEnv *env;
    ANativeActivity *activity = app_->activity;
    activity->vm->GetEnv((void **) &env, JNI_VERSION_1_6);

    activity->vm->AttachCurrentThread(&env, nullptr);

    jobject activityObj = env->NewGlobalRef(activity->clazz);
    jclass clz = env->GetObjectClass(activityObj);
    jint newOrientation = env->CallIntMethod(
            activityObj, env->GetMethodID(clz, "getRotationDegree", "()I"));
    env->DeleteGlobalRef(activityObj);

    activity->vm->DetachCurrentThread();
    return newOrientation;
}

std::string SudokuApplication::jniGetModelPath() {


    ASSERT(app_, "Application is not initialized");

    JNIEnv *env;
    ANativeActivity *activity = app_->activity;
    activity->vm->GetEnv((void **) &env, JNI_VERSION_1_6);

    activity->vm->AttachCurrentThread(&env, nullptr);

    jobject activityObj = env->NewGlobalRef(activity->clazz);
    jclass clz = env->GetObjectClass(activityObj);
    auto jmodel_path = (jstring) (env->CallObjectMethod(
            activityObj, env->GetMethodID(clz, "getModelPath", "()Ljava/lang/String;")));
    env->DeleteGlobalRef(activityObj);

    std::string model_path = jstring2string(env, jmodel_path);

    activity->vm->DetachCurrentThread();
    return model_path;
}

void SudokuApplication::jniUpdateUI() {
    JNIEnv *jni;
    app_->activity->vm->AttachCurrentThread(&jni, nullptr);

    jclass clazz = jni->GetObjectClass(app_->activity->clazz);
    jmethodID methodID = jni->GetMethodID(clazz, "updateUI", "()V");
    ASSERT(methodID, "JavaUI interface Object failed(%p)", methodID);

    jni->CallVoidMethod(app_->activity->clazz, methodID);
    app_->activity->vm->DetachCurrentThread();
}

void SudokuApplication::jniOnPhotoTaken(const char *fileName) {
    JNIEnv *jni;
    app_->activity->vm->AttachCurrentThread(&jni, nullptr);

    // Default class retrieval
    jclass clazz = jni->GetObjectClass(app_->activity->clazz);
    jmethodID methodID = jni->GetMethodID(clazz, "onPhotoTaken", "(Ljava/lang/String;)V");
    jstring javaName = jni->NewStringUTF(fileName);

    jni->CallVoidMethod(app_->activity->clazz, methodID, javaName);
    app_->activity->vm->DetachCurrentThread();
}

extern "C" JNIEXPORT void JNICALL
Java_com_jbamberger_sudoku4android_NativeSudokuActivity_notifyCameraPermission(
        JNIEnv *env, jclass type, jboolean permission) {
    std::thread permissionHandler(
            &SudokuApplication::onCameraPermission, getApplication(), permission != JNI_FALSE);
    permissionHandler.detach();
}

extern "C" JNIEXPORT void JNICALL
Java_com_jbamberger_sudoku4android_NativeSudokuActivity_takePhoto(
        JNIEnv *env, jclass type) {
    std::thread takePhotoHandler(&SudokuApplication::onTakePhoto, getApplication());
    takePhotoHandler.detach();
}
