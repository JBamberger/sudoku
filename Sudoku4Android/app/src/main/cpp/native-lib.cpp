#include <jni.h>
#include <string>
#include <opencv2/core.hpp>

float computeOpenCV();


extern "C" JNIEXPORT jstring JNICALL
Java_com_jbamberger_sudoku4android_MainActivity_stringFromJNI(
        JNIEnv *env,
        jobject /* this */) {

    auto x = computeOpenCV();

    std::stringstream builder;
    builder << "Hello from C++" << std::endl << x << std::endl;


    return env->NewStringUTF(builder.str().c_str());
}


float computeOpenCV() {
    auto mat = cv::Mat(5, 5, CV_32F);
    cv::setIdentity(mat);
    cv::Mat n = mat + mat * mat;

    float result = n.at<float>(2, 2);
    return result;
}