
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
    int64_t range[3];

    // Default class retrieval
    jclass clazz = jni->GetObjectClass(app_->activity->clazz);
    jmethodID methodID = jni->GetMethodID(clazz, "updateUI", "([J)V");

    // Parameter for updateUI: Semantics:  [exposure min, max, val, sensitivity min, max, val]
    jlongArray initData = jni->NewLongArray(6);

    ASSERT(initData && methodID, "JavaUI interface Object failed(%p, %p)", methodID, initData);

    if (!camera_->GetExposureRange(&range[0], &range[1], &range[2])) {
        memset(range, 0, sizeof(int64_t) * 3);
    }

    jni->SetLongArrayRegion(initData, 0, 3, range);

    if (!camera_->GetSensitivityRange(&range[0], &range[1], &range[2])) {
        memset(range, 0, sizeof(int64_t) * 3);
    }
    jni->SetLongArrayRegion(initData, 3, 3, range);

    jni->CallVoidMethod(app_->activity->clazz, methodID, initData);
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

extern "C" JNIEXPORT void JNICALL
Java_com_jbamberger_sudoku4android_NativeSudokuActivity_onExposureChanged(
        JNIEnv *env, jobject instance, jlong exposurePercent) {
    getApplication()->onCameraParameterChanged(ACAMERA_SENSOR_EXPOSURE_TIME, exposurePercent);
}

extern "C" JNIEXPORT void JNICALL
Java_com_jbamberger_sudoku4android_NativeSudokuActivity_onSensitivityChanged(
        JNIEnv *env, jobject instance, jlong sensitivity) {
    getApplication()->onCameraParameterChanged(ACAMERA_SENSOR_SENSITIVITY, sensitivity);
}
