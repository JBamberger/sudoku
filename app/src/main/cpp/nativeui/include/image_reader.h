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

#ifndef CAMERA_IMAGE_READER_H
#define CAMERA_IMAGE_READER_H

#include <media/NdkImageReader.h>
#include <functional>
#include <opencv2/core/types.hpp>

/*
 * ImageFormat:
 *     A Data Structure to communicate resolution between camera and ImageReader
 */
struct ImageFormat {
    int32_t width;
    int32_t height;
    int32_t format;  // Through out this demo, the format is fixed to YUV_420 format
};

struct ImagePlane {
    int32_t length;
    uint8_t *data;
    int32_t rowStride;
    int32_t pixelStride;
};

class Image {
private:
    AImage *image_;
public:
    explicit Image(AImage *image);

    ~Image();

    cv::Rect cropRect() const;

    int32_t getFormat() const;

    int32_t getNumPlanes() const;

    cv::Size size() const;

    ImagePlane getImagePlane(int planeIdx) const;
};

class ImageReader {
public:
    explicit ImageReader(ImageFormat *res, enum AIMAGE_FORMATS format);

    ~ImageReader();

    /**
     * Report cached ANativeWindow, which was used to create camera's capture
     * session output.
     */
    ANativeWindow *GetNativeWindow();

    /**
     * Retrieve Image on the top of Reader's queue
     */
    std::unique_ptr<Image> GetNextImage();

    /**
    * Retrieve Image on the back of Reader's queue, dropping older images
    */
    std::unique_ptr<Image> GetLatestImage();


    /**
     * AImageReader callback handler. Called by AImageReader when a frame is
     * captured
     * (Internal function, not to be called by clients)
     */
    void ImageCallback(AImageReader *reader);

    /**
     * DisplayImage()
     *   Present camera image to the given display buffer. Available image is converted
     *   to display buffer format. Supported display format:
     *      WINDOW_FORMAT_RGBX_8888
     *      WINDOW_FORMAT_RGBA_8888
     *   @param buf {@link ANativeWindow_Buffer} for image to display to.
     *   @param image
     */
    bool DisplayImage(ANativeWindow_Buffer *buf, std::unique_ptr<Image> image) const;

    /**
     * Configure the rotation angle necessary to apply to
     * Camera image when presenting: all rotations should be accumulated:
     *    CameraSensorOrientation + Android Device Native Orientation +
     *    Human Rotation (rotated degree related to Phone native orientation
     */
    void SetPresentRotation(int32_t angle);

    /**
     * regsiter a callback function for client to be notified that jpeg already
     * written out.
     * @param ctx is client context when callback is invoked
     * @param callback is the actual callback function
     */
    void RegisterCallback(void *ctx, std::function<void(void *ctx, const char *fileName)>);

private:
    int32_t presentRotation_;
    AImageReader *reader_;

    std::function<void(void *ctx, const char *fileName)> callback_;
    void *callbackCtx_;

    void WriteFile(AImage *image);
};

#endif  // CAMERA_IMAGE_READER_H
