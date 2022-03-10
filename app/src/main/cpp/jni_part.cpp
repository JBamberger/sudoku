#include <jni.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <vector>
#include <SudokuDetector.h>

void cvCompute(const cv::Mat &mGr, cv::Mat &mRgb);

std::string jstring2string(JNIEnv *env, jstring jStr);

std::unique_ptr<SudokuDetector> sudokuDetector;

extern "C"
JNIEXPORT void JNICALL
Java_com_jbamberger_sudoku4android_MainCameraActivity_init(JNIEnv *env, jobject thiz,
                                                           jstring model_path) {

    sudokuDetector = std::make_unique<SudokuDetector>(jstring2string(env, model_path));
}

extern "C" JNIEXPORT void JNICALL
Java_com_jbamberger_sudoku4android_MainCameraActivity_FindFeatures(JNIEnv *, jobject,
                                                                   jlong addrGray,
                                                                   jlong addrRgba) {
    cv::Mat &mGr = *(cv::Mat *) addrGray;
    cv::Mat &mRgb = *(cv::Mat *) addrRgba;

//    cvCompute(mGr, mRgb);

    const auto detection = sudokuDetector->detect(mRgb);
    if (detection->foundSudoku) {
        detection->drawOverlay(mRgb);
    }
}


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

void cvCompute(const cv::Mat &mGr, cv::Mat &mRgb) {
    std::vector<cv::KeyPoint> keypoints;
    cv::Ptr<cv::FeatureDetector> detector = cv::FastFeatureDetector::create(50);
    detector->detect(mGr, keypoints);
    for (const cv::KeyPoint &kp : keypoints) {
        circle(mRgb, cv::Point(kp.pt.x, kp.pt.y), 10, cv::Scalar(255, 0, 0, 255));
    }
}

