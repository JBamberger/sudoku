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

#include <utility>
#include <queue>
#include <unistd.h>
#include <cinttypes>
#include <camera/NdkCameraManager.h>
#include "camera_manager.h"
#include "native_debug.h"
#include "camera_utils.h"

NDKCamera::NDKCamera()
        : cameraMgr_(nullptr),
          activeCameraId_(""),
          cameraFacing_(ACAMERA_LENS_FACING_BACK),
          cameraOrientation_(0),
          outputContainer_(nullptr),
          captureSessionState_(CaptureSessionState::MAX_STATE) {

    valid_ = false;
    requests_.resize(CAPTURE_REQUEST_COUNT);
    memset(requests_.data(), 0, requests_.size() * sizeof(requests_[0]));
    cameras_.clear();
    cameraMgr_ = ACameraManager_create();
    ASSERT(cameraMgr_, "Failed to create cameraManager");

    // Pick up a back-facing camera to preview
    EnumerateCamera();
    ASSERT(activeCameraId_.size(), "Unknown ActiveCameraIdx");

    CALL_MGR(openCamera(cameraMgr_, activeCameraId_.c_str(), GetDeviceListener(),
                        &cameras_[activeCameraId_].device_));

    CALL_MGR(registerAvailabilityCallback(cameraMgr_, GetManagerListener()));

    valid_ = true;
}

/**
 * A helper class to assist image size comparison, by comparing the absolute
 * size regardless of the portrait or landscape mode.
 */
class DisplayDimension {
public:
    DisplayDimension(int32_t w, int32_t h) : w_(w), h_(h), portrait_(false) {
        if (h > w) {
            // make it landscape
            w_ = h;
            h_ = w;
            portrait_ = true;
        }
    }

    DisplayDimension(const DisplayDimension &other) {
        w_ = other.w_;
        h_ = other.h_;
        portrait_ = other.portrait_;
    }

    DisplayDimension() {
        w_ = 0;
        h_ = 0;
        portrait_ = false;
    }

    [[nodiscard]]  DisplayDimension &operator=(const DisplayDimension &other) = default;

    [[nodiscard]] bool isSameRatio(const DisplayDimension &other) const {
        return (w_ * other.h_ == h_ * other.w_);
    }

    [[nodiscard]] bool operator>(const DisplayDimension &other) const {
        return (w_ >= other.w_ & h_ >= other.h_);
    }

    [[nodiscard]] bool operator==(const DisplayDimension &other) const {
        return (w_ == other.w_ && h_ == other.h_ && portrait_ == other.portrait_);
    }

    [[nodiscard]] DisplayDimension operator-(const DisplayDimension &other) const {
        DisplayDimension delta(w_ - other.w_, h_ - other.h_);
        return delta;
    }

    void flip() { portrait_ = !portrait_; }

    [[nodiscard]] bool isPortrait() const { return portrait_; }

    [[nodiscard]] int32_t width() const { return w_; }

    [[nodiscard]] int32_t height() const { return h_; }

    [[nodiscard]] int32_t org_width() const { return (portrait_ ? h_ : w_); }

    [[nodiscard]] int32_t org_height() const { return (portrait_ ? w_ : h_); }

private:
    int32_t w_, h_;
    bool portrait_;
};


bool NDKCamera::MatchCaptureSizeRequest(
        ANativeWindow *display, ImageFormat *resView, ImageFormat *resCap) {

    int32_t requestWidth = ANativeWindow_getWidth(display);
    int32_t requestHeight = ANativeWindow_getHeight(display);
    DisplayDimension disp(requestWidth, requestHeight);
    if (cameraOrientation_ == 90 || cameraOrientation_ == 270) {
        disp.flip();
    }

    ACameraMetadata *metadata;
    CALL_MGR(getCameraCharacteristics(cameraMgr_, activeCameraId_.c_str(), &metadata));
    ACameraMetadata_const_entry entry;
    CALL_METADATA(getConstEntry(metadata, ACAMERA_SCALER_AVAILABLE_STREAM_CONFIGURATIONS, &entry));
    // format of the data: format, width, height, input?, type int32
    bool foundIt = false;
    DisplayDimension foundRes(4000, 4000);
    DisplayDimension maxJpg(0, 0);

    for (int i = 0; i < entry.count; i += 4) {
        int32_t input = entry.data.i32[i + 3];
        int32_t format = entry.data.i32[i + 0];
        if (input) continue;

        if (format == AIMAGE_FORMAT_YUV_420_888 || format == AIMAGE_FORMAT_JPEG) {
            DisplayDimension res(entry.data.i32[i + 1], entry.data.i32[i + 2]);

            if (!disp.isSameRatio(res)) continue;

            if (format == AIMAGE_FORMAT_YUV_420_888 && foundRes > res) {
                foundIt = true;
                foundRes = res;
            } else if (format == AIMAGE_FORMAT_JPEG && res > maxJpg) {
                maxJpg = res;
            }
        }
    }

    if (foundIt) {
        resView->width = foundRes.org_width();
        resView->height = foundRes.org_height();
        if (resCap) {
            resCap->width = maxJpg.org_width();
            resCap->height = maxJpg.org_height();
        }
    } else {
        LOGW("Did not find any compatible camera resolution, taking 640x480");
        if (disp.isPortrait()) {
            resView->width = 480;
            resView->height = 640;
        } else {
            resView->width = 640;
            resView->height = 480;
        }
        if (resCap)
            *resCap = *resView;
    }
    resView->format = AIMAGE_FORMAT_YUV_420_888;
    if (resCap) resCap->format = AIMAGE_FORMAT_JPEG;
    return foundIt;
}


void NDKCamera::CreateSession(
        ANativeWindow *previewWindow, ANativeWindow *jpgWindow, int32_t imageRotation) {

    requests_[PREVIEW_REQUEST_IDX].outputNativeWindow_ = previewWindow;
    requests_[PREVIEW_REQUEST_IDX].template_ = TEMPLATE_PREVIEW;

    requests_[JPG_CAPTURE_REQUEST_IDX].outputNativeWindow_ = jpgWindow;
    requests_[JPG_CAPTURE_REQUEST_IDX].template_ = TEMPLATE_STILL_CAPTURE;

    CALL_CONTAINER(create(&outputContainer_));
    for (auto &req : requests_) {
        if (!req.outputNativeWindow_) continue;

        ANativeWindow_acquire(req.outputNativeWindow_);
        CALL_OUTPUT(create(req.outputNativeWindow_, &req.sessionOutput_));
        CALL_CONTAINER(add(outputContainer_, req.sessionOutput_));
        CALL_TARGET(create(req.outputNativeWindow_, &req.target_));
        CALL_DEV(createCaptureRequest(
                cameras_[activeCameraId_].device_, req.template_, &req.request_));
        CALL_REQUEST(addTarget(req.request_, req.target_));
    }

    // Create a capture session for the given preview request
    captureSessionState_ = CaptureSessionState::READY;
    CALL_DEV(createCaptureSession(cameras_[activeCameraId_].device_,
                                  outputContainer_, GetSessionListener(),
                                  &captureSession_));

    if (jpgWindow) {
        ACaptureRequest_setEntry_i32(requests_[JPG_CAPTURE_REQUEST_IDX].request_,
                                     ACAMERA_JPEG_ORIENTATION, 1, &imageRotation);
    }
}

NDKCamera::~NDKCamera() {
    valid_ = false;
    // stop session if it is on:
    if (captureSessionState_ == CaptureSessionState::ACTIVE) {
        ACameraCaptureSession_stopRepeating(captureSession_);
    }
    ACameraCaptureSession_close(captureSession_);

    for (auto &req : requests_) {
        if (!req.outputNativeWindow_) continue;

        CALL_REQUEST(removeTarget(req.request_, req.target_));
        ACaptureRequest_free(req.request_);
        ACameraOutputTarget_free(req.target_);

        CALL_CONTAINER(remove(outputContainer_, req.sessionOutput_));
        ACaptureSessionOutput_free(req.sessionOutput_);

        ANativeWindow_release(req.outputNativeWindow_);
    }

    requests_.resize(0);
    ACaptureSessionOutputContainer_free(outputContainer_);

    for (auto &cam : cameras_) {
        if (cam.second.device_) {
            CALL_DEV(close(cam.second.device_));
        }
    }
    cameras_.clear();
    if (cameraMgr_) {
        CALL_MGR(unregisterAvailabilityCallback(cameraMgr_, GetManagerListener()));
        ACameraManager_delete(cameraMgr_);
        cameraMgr_ = nullptr;
    }
}

/**
 * EnumerateCamera()
 *     Loop through cameras on the system, pick up
 *     1) back facing one if available
 *     2) otherwise pick the first one reported to us
 */
void NDKCamera::EnumerateCamera() {
    ACameraIdList *cameraIds = nullptr;
    CALL_MGR(getCameraIdList(cameraMgr_, &cameraIds));

    for (int i = 0; i < cameraIds->numCameras; ++i) {
        const char *id = cameraIds->cameraIds[i];

        ACameraMetadata *metadataObj;
        CALL_MGR(getCameraCharacteristics(cameraMgr_, id, &metadataObj));

        int32_t count = 0;
        const uint32_t *tags = nullptr;
        ACameraMetadata_getAllTags(metadataObj, &count, &tags);
        for (int tagIdx = 0; tagIdx < count; ++tagIdx) {
            if (ACAMERA_LENS_FACING == tags[tagIdx]) {
                ACameraMetadata_const_entry lensInfo = {0,};
                CALL_METADATA(getConstEntry(metadataObj, tags[tagIdx], &lensInfo));
                CameraId cam(id);
                cam.facing_ = static_cast<acamera_metadata_enum_android_lens_facing_t>(
                        lensInfo.data.u8[0]);
                cam.owner_ = false;
                cam.device_ = nullptr;
                cameras_[cam.id_] = cam;
                if (cam.facing_ == ACAMERA_LENS_FACING_BACK) {
                    activeCameraId_ = cam.id_;
                }
                break;
            }
        }
        ACameraMetadata_free(metadataObj);
    }

    ASSERT(cameras_.size(), "No Camera Available on the device");
    if (activeCameraId_.length() == 0) {
        // if no back facing camera found, pick up the first one to use...
        activeCameraId_ = cameras_.begin()->second.id_;
    }
    ACameraManager_deleteCameraIdList(cameraIds);
}

bool NDKCamera::GetSensorOrientation(uint8_t *facing, int32_t *angle) {
    if (!cameraMgr_) {
        return false;
    }

    ACameraMetadata *metadataObj;
    CALL_MGR(getCameraCharacteristics(cameraMgr_, activeCameraId_.c_str(), &metadataObj));

    ACameraMetadata_const_entry face;
    CALL_METADATA(getConstEntry(metadataObj, ACAMERA_LENS_FACING, &face));
    cameraFacing_ = face.data.u8[0];

    ACameraMetadata_const_entry orientation;
    CALL_METADATA(getConstEntry(metadataObj, ACAMERA_SENSOR_ORIENTATION, &orientation));
    cameraOrientation_ = orientation.data.i32[0];

    ACameraMetadata_free(metadataObj);

    if (facing) *facing = cameraFacing_;
    if (angle) *angle = cameraOrientation_;
    return true;
}

void NDKCamera::StartPreview(bool start) {
    if (start) {
        CALL_SESSION(setRepeatingRequest(captureSession_, nullptr, 1,
                                         &requests_[PREVIEW_REQUEST_IDX].request_,
                                         nullptr));
    } else if (captureSessionState_ == CaptureSessionState::ACTIVE) {
        ACameraCaptureSession_stopRepeating(captureSession_);
    } else {
        ASSERT(false, "No active session to stop! SessionState=%d)", captureSessionState_);
    }
}

bool NDKCamera::TakePhoto() {
    if (captureSessionState_ == CaptureSessionState::ACTIVE) {
        ACameraCaptureSession_stopRepeating(captureSession_);
    }

    CALL_SESSION(capture(captureSession_, GetCaptureCallback(), 1,
                         &requests_[JPG_CAPTURE_REQUEST_IDX].request_,
                         &requests_[JPG_CAPTURE_REQUEST_IDX].sessionSequenceId_));
    return true;
}