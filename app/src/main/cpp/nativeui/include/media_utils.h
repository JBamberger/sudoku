//
// Created by jannik on 17.03.2022.
//

#ifndef SUDOKU4ANDROID_MEDIA_UTILS_H
#define SUDOKU4ANDROID_MEDIA_UTILS_H

#include <media/NdkImage.h>
#include <media/NdkMediaError.h>

#define CALL_MEDIA(func)                                             \
  {                                                                  \
    media_status_t status = func;                                    \
    ASSERT(status == AMEDIA_OK, "%s call failed with code: %#x, %s", \
           __FUNCTION__, status, GetErrorStr(status));               \
  }
#define CALL_IMAGE(func) CALL_MEDIA(AImage_##func)

const char *GetErrorStr(media_status_t err);

#endif //SUDOKU4ANDROID_MEDIA_UTILS_H
