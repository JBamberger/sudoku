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

/**
 * Constructor
 */
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
        std::thread writeFileHandler(&ImageReader::WriteFile, this, image);
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

/**
 * GetNextImage()
 *   Retrieve the next image in ImageReader's bufferQueue, NOT the last image so
 * no image is skipped. Recommended for batch/background processing.
 */
AImage *ImageReader::GetNextImage() {
    AImage *image;
    media_status_t status = AImageReader_acquireNextImage(reader_, &image);
    if (status != AMEDIA_OK) {
        return nullptr;
    }
    return image;
}

/**
 * GetLatestImage()
 *   Retrieve the last image in ImageReader's bufferQueue, deleting images in
 * in front of it on the queue. Recommended for real-time processing.
 */
AImage *ImageReader::GetLatestImage() {
    AImage *image;
    media_status_t status = AImageReader_acquireLatestImage(reader_, &image);
    if (status != AMEDIA_OK) {
        return nullptr;
    }
    return image;
}

/**
 * Delete Image
 * @param image {@link AImage} instance to be deleted
 */
void ImageReader::DeleteImage(AImage *image) {
    if (image) AImage_delete(image);
}


cv::Mat copyImageToSurface(ANativeWindow_Buffer *buf, AImage *image, int32_t rotation) {
    /// y luminance
    /// u chroma (blue projection)
    /// v chroma (red projection)

    // Wrapper to access the native window buffer
    auto rgbOutput = cv::Mat(buf->height, buf->stride, CV_8UC4, buf->bits);

    // Region of the image that can be used for processing
    AImageCropRect srcRect;
    AImage_getCropRect(image, &srcRect);
    cv::Rect yRect(cv::Point(srcRect.left, srcRect.top),
                   cv::Point(srcRect.right, srcRect.bottom));
    cv::Rect uvRect(cv::Point(srcRect.left / 2, srcRect.top / 2),
                    cv::Point(srcRect.right / 2, srcRect.bottom / 2));

    int32_t srcFormat = -1;
    AImage_getFormat(image, &srcFormat);
    ASSERT(AIMAGE_FORMAT_YUV_420_888 == srcFormat, "Failed to get format");

    int32_t numPlanes = 0;
    AImage_getNumberOfPlanes(image, &numPlanes);
    ASSERT(numPlanes == 3, "Image must have 3 planes. Has %d", numPlanes);

    int32_t h = -1, w = -1;
    AImage_getWidth(image, &h);
    AImage_getWidth(image, &w);

    int32_t yLen, uLen, vLen;
    uint8_t *yPixel, *uPixel, *vPixel;
    AImage_getPlaneData(image, 0, &yPixel, &yLen);
    AImage_getPlaneData(image, 1, &vPixel, &vLen);
    AImage_getPlaneData(image, 2, &uPixel, &uLen);

    int32_t yRowStride, uRowStride, vRowStride;
    AImage_getPlaneRowStride(image, 0, &yRowStride);
    AImage_getPlaneRowStride(image, 1, &uRowStride);
    AImage_getPlaneRowStride(image, 2, &vRowStride);

    int32_t yPixelStride, uPixelStride, vPixelStride;
    AImage_getPlanePixelStride(image, 0, &yPixelStride);
    AImage_getPlanePixelStride(image, 1, &uPixelStride);
    AImage_getPlanePixelStride(image, 2, &vPixelStride);

    LOGI("Output: Format %d h=%d, w=%d, stride=%d len=%d",
         buf->format, buf->height, buf->width, buf->stride, buf->height * buf->width);
    LOGI("Image:  Format %d, numPlanes %d, h=%d, w=%d, len=%d", srcFormat, numPlanes, h, w, h * w);
    LOGI("Rect: %d %d %d %d", srcRect.left, srcRect.right, srcRect.top, srcRect.bottom);
    LOGI("Length:    %20d %20d %20d %20d", yLen, uLen, vLen, yLen + uLen + vLen);
    LOGI("Ptr:       %20p %20p %20p", (void *) yPixel, (void *) uPixel, (void *) vPixel);
    LOGI("RowStride: %20d %20d %20d", yRowStride, uRowStride, vRowStride);
    LOGI("PixStride: %20d %20d %20d", yPixelStride, uPixelStride, vPixelStride);


    // plane #0 Y
    // plane #1 U (Cb)
    // plane #2 V (Cr)

    // Y not interleaved with U/V, stride==1
    ASSERT(yPixelStride == 1, "Luminance (Y) stride must be 1.");
    // U stride == V stride
    ASSERT(uPixelStride == vPixelStride, "Chroma pixel strides must be equal.")
    ASSERT(uRowStride == vRowStride, "Chroma row strides must be equal.")

    cv::Mat rgbNoRot;
    if (uPixelStride == 1) { // Chroma channels are not interleaved
        auto copyPlane = [](
                const uint8_t *const plane, int32_t stride, const cv::Rect &bounds, uint8_t *out) {

            if (bounds.x == 0 && bounds.width == stride) {
                memcpy(out, plane + bounds.y * stride, bounds.height * bounds.width);
            } else {
                for (int i = bounds.y; i < bounds.y + bounds.height; i++) {
                    memcpy(out, plane + i * stride, bounds.width);
                }
            }
        };

        auto ySize = yRect.height * yRect.width;
        auto uvSize = uvRect.height * uvRect.width;

        cv::Mat yuvMat(yRect.height + yRect.height / 2, yRect.width, CV_8UC1);
        uint8_t *yuvPtr = yuvMat.data;
        copyPlane(yPixel, yRowStride, yRect, yuvPtr);
        yuvPtr += ySize;
        copyPlane(uPixel, uRowStride, uvRect, yuvPtr);
        yuvPtr += uvSize;
        copyPlane(vPixel, vRowStride, uvRect, yuvPtr);

        cv::cvtColor(yuvMat, rgbNoRot, cv::COLOR_YUV2BGRA_I420);
    } else if (uPixelStride == 2) { // Chroma channels are interleaved
        cv::Mat y_mat(h, w, CV_8UC1, yPixel, yRowStride);
        y_mat = y_mat(yRect);

        long addr_diff = vPixel - uPixel;
        if (addr_diff == 1) {
            cv::Mat uv_mat(h / 2, w / 2, CV_8UC2, uPixel, uRowStride);
            cv::cvtColorTwoPlane(y_mat, uv_mat(uvRect), rgbNoRot, cv::COLOR_YUV2RGBA_NV21);
        } else if (addr_diff == -1) {
            cv::Mat uv_mat(h / 2, w / 2, CV_8UC2, vPixel, vRowStride);
            cv::cvtColorTwoPlane(y_mat, uv_mat(uvRect), rgbNoRot, cv::COLOR_YUV2RGBA_NV12);
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
    switch (rotation) {
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
            ASSERT(0, "NOT recognized display rotation: %d", rotation);
    }

    rot.copyTo(rgbOutput({0, 0, rot.cols, rot.rows}));

    return rgbOutput;
}

/**
 * Convert yuv image inside AImage into ANativeWindow_Buffer ANativeWindow_Buffer format is
 * guaranteed to be WINDOW_FORMAT_RGBX_8888 or WINDOW_FORMAT_RGBA_8888.
 * @param buf a {@link ANativeWindow_Buffer } instance, destination of image conversion
 * @param image a {@link AImage} instance, source of image conversion.
 *            it will be deleted via {@link AImage_delete}
 */
bool ImageReader::DisplayImage(ANativeWindow_Buffer *buf, AImage *image) const {
    ASSERT(buf->format == WINDOW_FORMAT_RGBX_8888 ||
           buf->format == WINDOW_FORMAT_RGBA_8888,
           "Not supported buffer format");

    int32_t srcFormat = -1;
    AImage_getFormat(image, &srcFormat);
    ASSERT(AIMAGE_FORMAT_YUV_420_888 == srcFormat, "Failed to get format");

    int32_t srcPlanes = 0;
    AImage_getNumberOfPlanes(image, &srcPlanes);
    ASSERT(srcPlanes == 3, "Is not 3 planes");


    copyImageToSurface(buf, image, presentRotation_);

    AImage_delete(image);
    return true;
}

void ImageReader::SetPresentRotation(int32_t angle) {
    presentRotation_ = angle;
}

/**
 * Write out jpeg files to kDirName directory
 * @param image point capture jpg image
 */
void ImageReader::WriteFile(AImage *image) {

    int planeCount;
    media_status_t status = AImage_getNumberOfPlanes(image, &planeCount);
    ASSERT(status == AMEDIA_OK && planeCount == 1,
           "Error: getNumberOfPlanes() planeCount = %d", planeCount);

    uint8_t *data = nullptr;
    int len = 0;
    AImage_getPlaneData(image, 0, &data, &len);

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
    if (file && data && len) {
        fwrite(data, 1, len, file);
        fclose(file);

        if (callback_) {
            callback_(callbackCtx_, fileName.c_str());
        }
    } else {
        if (file)
            fclose(file);
    }
    AImage_delete(image);
}
