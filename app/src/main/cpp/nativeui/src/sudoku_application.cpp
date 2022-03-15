#include <cstdio>
#include "sudoku_application.h"
#include "native_debug.h"

SudokuApplication::SudokuApplication(android_app *app)
        : app_(app),
          cameraGranted_(false),
          rotation_(0),
          cameraReady_(false),
          camera_(nullptr),
          yuvReader_(nullptr),
          jpgReader_(nullptr) {
    memset(&savedNativeWinRes_, 0, sizeof(savedNativeWinRes_));
}

SudokuApplication::~SudokuApplication() {
    cameraReady_ = false;
    deleteCamera();
}

void SudokuApplication::createCamera() {
    if (!cameraGranted_ || !app_->window) {
        LOGW("Permissions are missing or rendering surface is not initialized yet.");
        return;
    }

    rotation_ = jniGetDisplayRotation();

    camera_ = new NDKCamera();
    ASSERT(camera_, "Failed to Create CameraObject");

    int32_t imageRotation = computeImageRotation();

    ImageFormat view{0, 0, 0}, capture{0, 0, 0};
    camera_->MatchCaptureSizeRequest(app_->window, &view, &capture);
    ASSERT(view.width && view.height, "Could not find supportable resolution");

    bool portraitNativeWindow = (savedNativeWinRes_.width < savedNativeWinRes_.height);
    ANativeWindow_setBuffersGeometry(
            app_->window,
            portraitNativeWindow ? view.height : view.width,
            portraitNativeWindow ? view.width : view.height,
            WINDOW_FORMAT_RGBA_8888);

    yuvReader_ = new ImageReader(&view, AIMAGE_FORMAT_YUV_420_888);
    yuvReader_->SetPresentRotation(imageRotation);

    jpgReader_ = new ImageReader(&capture, AIMAGE_FORMAT_JPEG);
    jpgReader_->SetPresentRotation(imageRotation);
    jpgReader_->RegisterCallback(this, [](void *ctx, const char *str) -> void {
        reinterpret_cast<SudokuApplication * >(ctx)->jniOnPhotoTaken(str);
    });

    camera_->CreateSession(
            yuvReader_->GetNativeWindow(),
            jpgReader_->GetNativeWindow(),
            imageRotation);
}

int32_t SudokuApplication::computeImageRotation() {
    int32_t facing = 0, angle = 0, imageRotation = 0;
    if (camera_->GetSensorOrientation(&facing, &angle)) {
        if (facing == ACAMERA_LENS_FACING_FRONT) {
            imageRotation = (angle + rotation_) % 360;
            imageRotation = (360 - imageRotation) % 360;
        } else {
            imageRotation = (angle - rotation_ + 360) % 360;
        }
    }
    LOGI("Phone Rotation: %d, Present Rotation Angle: %d", rotation_, imageRotation);
    return imageRotation;
}

void SudokuApplication::deleteCamera() {
    cameraReady_ = false;
    if (camera_) {
        delete camera_;
        camera_ = nullptr;
    }
    if (yuvReader_) {
        delete yuvReader_;
        yuvReader_ = nullptr;
    }
    if (jpgReader_) {
        delete jpgReader_;
        jpgReader_ = nullptr;
    }
}

void SudokuApplication::onTakePhoto() {
    if (camera_) {
        camera_->TakePhoto();
    } else {
        LOGW("Attempted to take photo but camera is not initialized.");
    }
}

void SudokuApplication::onCameraPermission(bool granted) {
    cameraGranted_ = granted;

    if (cameraGranted_) {
        onAppInitWindow();
    }else {
        LOGW("Permissions were not granted!");
    }
}

void SudokuApplication::onCameraParameterChanged(int32_t code, int64_t val) {
    camera_->UpdateCameraRequestParameter(code, val);
}

/**
 * Convert yuv to RGBA8888 and render frame.
 */
void SudokuApplication::pollAndDrawFrame() {
    if (!cameraReady_ || !yuvReader_) return;

    AImage *image = yuvReader_->GetLatestImage();
    if (!image) return;

    ANativeWindow_acquire(app_->window);
    ANativeWindow_Buffer buf;
    if (ANativeWindow_lock(app_->window, &buf, nullptr) < 0) {
        ImageReader::DeleteImage(image);
        return;
    }

    // Converts YUV to RGBA8888 and outputs the image in the preview buffer
    yuvReader_->DisplayImage(&buf, image);

//    // Wrap output buffer with cv::Mat to allow access from OpenCV
//    auto &display_mat = cv::Mat(buffer.height, buffer.stride, CV_8UC4, buffer.bits);
//
//    // TODO: perform cv processing

    ANativeWindow_unlockAndPost(app_->window);
    ANativeWindow_release(app_->window);
}
