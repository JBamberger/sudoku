
#include "sudoku_application.h"
#include <native_debug.h>

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
