//
//  The following code has been taken from [1] to avoid the having to compile the entire opencv contrib module. The
//  code has been stripped down to include only code paths for the use case at hand. The following license agreement
//  applies:
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2014, Beat Kueng (beat-kueng@gmx.net), Lukas Vogel, Morten Lysgaard
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//  [1]: https://github.com/opencv/opencv_contrib/blob/master/modules/ximgproc/src/niblack_thresholding.cpp
//

#ifndef XIMGPROC_COMPAT_H
#define XIMGPROC_COMPAT_H

#include <opencv2/core.hpp>

inline void
savoulaThreshInv(const cv::Mat& src, cv::Mat& dst, double maxValue, int blockSize, double k)
{
    // Input grayscale image
    CV_Assert(src.channels() == 1);
    CV_Assert(blockSize % 2 == 1 && blockSize > 1);
    CV_Assert(src.depth() == CV_8U);

    // Compute local threshold (T = mean + k * stddev)
    // using mean and standard deviation in the neighborhood of each pixel
    // (intermediate calculations are done with floating-point precision)
    cv::Mat thresh;
    {
        // note that: Var[X] = E[X^2] - E[X]^2
        cv::Mat mean, sqmean, variance, stddev, sqrtVarianceMeanSum;
        boxFilter(src, mean, CV_32F, cv::Size(blockSize, blockSize), cv::Point(-1, -1), true, cv::BORDER_REPLICATE);
        sqrBoxFilter(
          src, sqmean, CV_32F, cv::Size(blockSize, blockSize), cv::Point(-1, -1), true, cv::BORDER_REPLICATE);
        variance = sqmean - mean.mul(mean);
        sqrt(variance, stddev);

        thresh = mean.mul(1. + static_cast<float>(k) * (stddev / 128 - 1.));
        thresh.convertTo(thresh, src.depth());
    }

    // Prepare output image
    dst.create(src.size(), src.type());
    CV_Assert(src.data != dst.data); // no inplace processing

    // Apply thresholding: ( pixel > threshold ) ? foreground : background
    cv::Mat mask;
    compare(src, thresh, mask, cv::CMP_LE);
    dst.setTo(0);
    dst.setTo(maxValue, mask);
}

#endif // XIMGPROC_COMPAT_H
