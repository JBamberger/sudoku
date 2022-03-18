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

#include <string>
#include <vector>
#include <functional>
#include <thread>
#include <cstdlib>
#include <dirent.h>
#include <ctime>
#include <utility>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgproc.hpp>
#include "image_reader.h"
#include "native_debug.h"
#include "media_utils.h"

/*
 * For JPEG capture, captured files are saved under
 *     DirName
 * File names are incrementally appended an index number as
 *     capture0.jpg, capture1.jpg, capture2.jpg
 */
static const char *kDirName = "/sdcard/DCIM/Camera/";
static const char *kFileName = "capture";

/**
 * MAX_BUF_COUNT:
 *   Max buffers in this ImageReader.
 */
#define MAX_BUF_COUNT 4

/**
 * ImageReader listener: called by AImageReader for every frame captured
 * We pass the event to ImageReader class, so it could do some housekeeping
 * about
 * the loaded queue. For example, we could keep a counter to track how many
 * buffers are full and idle in the queue. If camera almost has no buffer to
 * capture
 * we could release ( skip ) some frames by AImageReader_getNextImage() and
 * AImageReader_delete().
 */
void OnImageCallback(void *ctx, AImageReader *reader) {
    reinterpret_cast<ImageReader *>(ctx)->ImageCallback(reader);
}

Image::Image(AImage *image) : image_(image) {
}

Image::~Image() {
    if (image_) {
        AImage_delete(image_);
    }
}

cv::Rect Image::cropRect() const {
    AImageCropRect srcRect;
    CALL_IMAGE(getCropRect(image_, &srcRect));
    return cv::Rect(cv::Point(srcRect.left, srcRect.top),
                    cv::Point(srcRect.right, srcRect.bottom));
}

int32_t Image::getFormat() const {
    int32_t srcFormat = -1;
    CALL_IMAGE(getFormat(image_, &srcFormat));
    return srcFormat;
}

int32_t Image::getNumPlanes() const {
    int32_t numPlanes = 0;
    CALL_IMAGE(getNumberOfPlanes(image_, &numPlanes));
    return numPlanes;
}

cv::Size Image::size() const {
    int32_t h = -1, w = -1;
    CALL_IMAGE(getWidth(image_, &h));
    CALL_IMAGE(getWidth(image_, &w));
    return cv::Size(w, h);
}

ImagePlane Image::getImagePlane(int planeIdx) const {
    ImagePlane plane{};
    CALL_IMAGE(getPlaneData(image_, planeIdx, &plane.data, &plane.length));
    CALL_IMAGE(getPlaneRowStride(image_, planeIdx, &plane.rowStride));
    CALL_IMAGE(getPlanePixelStride(image_, planeIdx, &plane.pixelStride));
    return plane;
}

ImageReader::ImageReader(ImageFormat *res, enum AIMAGE_FORMATS format)
        : presentRotation_(0), reader_(nullptr) {
    callback_ = nullptr;
    callbackCtx_ = nullptr;

    media_status_t status = AImageReader_new(
            res->width, res->height, format, MAX_BUF_COUNT, &reader_);
    ASSERT(reader_ && status == AMEDIA_OK, "Failed to create AImageReader");

    AImageReader_ImageListener listener{
            .context = this,
            .onImageAvailable = OnImageCallback,
    };
    AImageReader_setImageListener(reader_, &listener);
}

ImageReader::~ImageReader() {
    ASSERT(reader_, "NULL Pointer to %s", __FUNCTION__);
    AImageReader_delete(reader_);
}

void ImageReader::RegisterCallback(
        void *ctx, std::function<void(void *ctx, const char *fileName)> func) {
    callbackCtx_ = ctx;
    callback_ = std::move(func);
}

void ImageReader::ImageCallback(AImageReader *reader) {
    int32_t format;
    {
        media_status_t status = AImageReader_getFormat(reader, &format);
        ASSERT(status == AMEDIA_OK, "Failed to get the media format");
    }

    if (format == AIMAGE_FORMAT_JPEG) {
        AImage *image = nullptr;
        {
            media_status_t status = AImageReader_acquireNextImage(reader, &image);
            ASSERT(status == AMEDIA_OK && image, "Image is not available");
        }

        // Create a thread and write out the jpeg files
        std::thread writeFileHandler(&ImageReader::WriteFile, this, std::make_unique<Image>(image));
        writeFileHandler.detach();
    }
}

ANativeWindow *ImageReader::GetNativeWindow() {
    if (!reader_) return nullptr;
    ANativeWindow *nativeWindow;
    media_status_t status = AImageReader_getWindow(reader_, &nativeWindow);
    ASSERT(status == AMEDIA_OK, "Could not get ANativeWindow");

    return nativeWindow;
}

std::unique_ptr<Image> ImageReader::GetNextImage() {
    AImage *image;
    media_status_t status = AImageReader_acquireNextImage(reader_, &image);
    if (status != AMEDIA_OK) {
        return nullptr;
    }
    return std::make_unique<Image>(image);
}

std::unique_ptr<Image> ImageReader::GetLatestImage() {
    AImage *image;
    media_status_t status = AImageReader_acquireLatestImage(reader_, &image);
    if (status != AMEDIA_OK) {
        return nullptr;
    }
    return std::make_unique<Image>(image);
}

bool ImageReader::DisplayImage(ANativeWindow_Buffer *buf, std::unique_ptr<Image> image) const {
    ASSERT(buf->format == WINDOW_FORMAT_RGBX_8888 ||
           buf->format == WINDOW_FORMAT_RGBA_8888,
           "Not supported buffer format");

    // Wrapper to access the native window buffer
    auto rgbOutput = cv::Mat(buf->height, buf->stride, CV_8UC4, buf->bits);

    // Region of the image that can be used for processing
    auto yRect = image->cropRect();
    cv::Rect uvRect(
            cv::Point(yRect.x / 2, yRect.y / 2),
            cv::Size(yRect.width / 2, yRect.height / 2));

    int32_t srcFormat = image->getFormat();
    ASSERT(AIMAGE_FORMAT_YUV_420_888 == srcFormat,
           "Invalid format. Must be %d but is %d.", AIMAGE_FORMAT_YUV_420_888, srcFormat);

    int32_t numPlanes = image->getNumPlanes();
    ASSERT(numPlanes == 3, "Image must have 3 planes. Has %d planes.", numPlanes);

    cv::Size imgSize = image->size();
    ImagePlane yPlane = image->getImagePlane(0);
    ImagePlane uPlane = image->getImagePlane(1);
    ImagePlane vPlane = image->getImagePlane(2);

    // plane0 Y, plane1 U (Cb), plane2 V (Cr)
    // Y not interleaved with U/V, stride==1
    ASSERT(yPlane.pixelStride == 1, "Luminance (Y) stride must be 1.");
    // U stride == V stride
    ASSERT(uPlane.pixelStride == vPlane.pixelStride, "Chroma pixel strides must be equal.")
    ASSERT(uPlane.rowStride == vPlane.rowStride, "Chroma row strides must be equal.")


//    LOGI("Output: Format %d h=%d, w=%d, stride=%d len=%d",
//         buf->format, buf->height, buf->width, buf->stride, buf->height * buf->width);
//    LOGI("Image: numPlanes %d, h=%d, w=%d, len=%d", numPlanes, h, w, h * w);
//    LOGI("Rect: %d %d %d %d", srcRect.left, srcRect.right, srcRect.top, srcRect.bottom);
//    LOGI("Length:    %20d %20d %20d %20d", yLen, uLen, vLen, yLen + uLen + vLen);
//    LOGI("Ptr:       %20p %20p %20p", (void *) yPixel, (void *) uPixel, (void *) vPixel);
//    LOGI("RowStride: %20d %20d %20d", yRowStride, uRowStride, vRowStride);
//    LOGI("PixStride: %20d %20d %20d", yPixelStride, uPixelStride, vPixelStride);

    cv::Mat rgbNoRot;
    if (uPlane.pixelStride == 1) { // Chroma channels are not interleaved
        auto copyPlane = [](const ImagePlane &p, const cv::Rect &bounds, uint8_t *out) {
            if (bounds.x == 0 && bounds.width == p.rowStride) {
                memcpy(out, p.data + bounds.y * p.rowStride, bounds.height * bounds.width);
            } else {
                for (int i = bounds.y; i < bounds.y + bounds.height; i++) {
                    memcpy(out, p.data + i * p.rowStride, bounds.width);
                }
            }
        };

        cv::Mat yuvMat(yRect.height + yRect.height / 2, yRect.width, CV_8UC1);
        uint8_t *yuvPtr = yuvMat.data;
        copyPlane(yPlane, yRect, yuvPtr);
        yuvPtr += yRect.area();
        copyPlane(uPlane, uvRect, yuvPtr);
        yuvPtr += uvRect.area();
        copyPlane(vPlane, uvRect, yuvPtr);

        cv::cvtColor(yuvMat, rgbNoRot, cv::COLOR_YUV2RGBA_I420);
    } else if (uPlane.pixelStride == 2) { // Chroma channels are interleaved
        cv::Mat yMat(imgSize.height, imgSize.width, CV_8UC1, yPlane.data, yPlane.rowStride);
        yMat = yMat(yRect);

        long addrDiff = vPlane.data - uPlane.data;
        if (addrDiff == 1) {
            cv::Mat uvMat(
                    imgSize.height / 2, imgSize.width / 2, CV_8UC2, uPlane.data, uPlane.rowStride);
            cv::cvtColorTwoPlane(yMat, uvMat(uvRect), rgbNoRot, cv::COLOR_YUV2RGBA_NV12);
        } else if (addrDiff == -1) {
            cv::Mat uvMat(
                    imgSize.height / 2, imgSize.width / 2, CV_8UC2, vPlane.data, vPlane.rowStride);
            cv::cvtColorTwoPlane(yMat, uvMat(uvRect), rgbNoRot, cv::COLOR_YUV2RGBA_NV21);
        } else {
            // TODO: Implement handling for interleaved data that does not share the same memory
            //  location.
            ASSERT(false, "Difference between u and v planes must be 1 or -1.");
        }
    } else {
        ASSERT(false, "Handling of this image format is not implemented!");
    }

    ASSERT(rgbNoRot.rows == yRect.height && rgbNoRot.cols == yRect.width,
           "Expected and actual output sizes differ. Expected: (h=%d, w=%d), Actual: (h=%d, w=%d)",
           rgbNoRot.rows, rgbNoRot.cols, yRect.height, yRect.width)

    cv::Mat rot;
    switch (presentRotation_) {
        case 0:
            rot = rgbNoRot;
            break;
        case 90:
            cv::rotate(rgbNoRot, rot, cv::ROTATE_90_CLOCKWISE);
            break;
        case 180:
            cv::rotate(rgbNoRot, rot, cv::ROTATE_180);
            break;
        case 270:
            cv::rotate(rgbNoRot, rot, cv::ROTATE_90_COUNTERCLOCKWISE);
            break;
        default:
            ASSERT(0, "NOT recognized display rotation: %d", presentRotation_);
    }

    rot.copyTo(rgbOutput({0, 0, rot.cols, rot.rows}));

    return true;
}

void ImageReader::SetPresentRotation(int32_t angle) {
    presentRotation_ = angle;
}

void ImageReader::WriteFile(std::unique_ptr<Image> image) {
    auto planeCount = image->getNumPlanes();
    ASSERT(planeCount == 1, "Error: getNumberOfPlanes() planeCount = %d", planeCount);

    auto plane = image->getImagePlane(0);

    DIR *dir = opendir(kDirName);
    if (dir) {
        closedir(dir);
    } else {
        std::string cmd = "mkdir -p ";
        cmd += kDirName;
        system(cmd.c_str());
    }

    struct timespec ts{0, 0};
    clock_gettime(CLOCK_REALTIME, &ts);
    struct tm localTime{};
    localtime_r(&ts.tv_sec, &localTime);

    std::string fileName = kDirName;
    std::string dash("-");
    fileName += kFileName + std::to_string(localTime.tm_mon) +
                std::to_string(localTime.tm_mday) + dash +
                std::to_string(localTime.tm_hour) +
                std::to_string(localTime.tm_min) +
                std::to_string(localTime.tm_sec) + ".jpg";
    FILE *file = fopen(fileName.c_str(), "wb");
    if (file && plane.data && plane.length) {
        fwrite(plane.data, 1, plane.length, file);
        fclose(file);

        if (callback_) {
            callback_(callbackCtx_, fileName.c_str());
        }
    } else {
        if (file)
            fclose(file);
    }
}
