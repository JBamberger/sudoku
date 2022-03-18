
#include <vector>
#include <string>
#include <utility>
#include "native_debug.h"
#include "media_utils.h"

#define UKNOWN_TAG "UNKNOW_TAG"
#define MAKE_PAIR(val) std::make_pair(val, #val)

template<typename T>
const char *GetPairStr(T key, std::vector<std::pair<T, const char *>> &store) {
    for (auto it = store.begin(); it != store.end(); ++it) {
        if (it->first == key) {
            return it->second;
        }
    }
    LOGW("(%#08x) : UNKNOWN_TAG for %s", key, typeid(store[0].first).name());
    return UKNOWN_TAG;
}

using ERROR_PAIR = std::pair<media_status_t, const char *>;
static std::vector<ERROR_PAIR> errorInfo{
        MAKE_PAIR(AMEDIA_OK),
        MAKE_PAIR(AMEDIACODEC_ERROR_INSUFFICIENT_RESOURCE),
        MAKE_PAIR(AMEDIACODEC_ERROR_RECLAIMED),
        MAKE_PAIR(AMEDIA_ERROR_BASE),
        MAKE_PAIR(AMEDIA_ERROR_UNKNOWN),
        MAKE_PAIR(AMEDIA_ERROR_MALFORMED),
        MAKE_PAIR(AMEDIA_ERROR_UNSUPPORTED),
        MAKE_PAIR(AMEDIA_ERROR_INVALID_OBJECT),
        MAKE_PAIR(AMEDIA_ERROR_INVALID_PARAMETER),
        MAKE_PAIR(AMEDIA_ERROR_INVALID_OPERATION),
        MAKE_PAIR(AMEDIA_ERROR_END_OF_STREAM),
        MAKE_PAIR(AMEDIA_ERROR_IO),
        MAKE_PAIR(AMEDIA_ERROR_WOULD_BLOCK),
        MAKE_PAIR(AMEDIA_DRM_ERROR_BASE),
        MAKE_PAIR(AMEDIA_DRM_NOT_PROVISIONED),
        MAKE_PAIR(AMEDIA_DRM_RESOURCE_BUSY),
        MAKE_PAIR(AMEDIA_DRM_DEVICE_REVOKED),
        MAKE_PAIR(AMEDIA_DRM_SHORT_BUFFER),
        MAKE_PAIR(AMEDIA_DRM_SESSION_NOT_OPENED),
        MAKE_PAIR(AMEDIA_DRM_TAMPER_DETECTED),
        MAKE_PAIR(AMEDIA_DRM_VERIFY_FAILED),
        MAKE_PAIR(AMEDIA_DRM_NEED_KEY),
        MAKE_PAIR(AMEDIA_DRM_LICENSE_EXPIRED),
        MAKE_PAIR(AMEDIA_IMGREADER_ERROR_BASE),
        MAKE_PAIR(AMEDIA_IMGREADER_NO_BUFFER_AVAILABLE),
        MAKE_PAIR(AMEDIA_IMGREADER_MAX_IMAGES_ACQUIRED),
        MAKE_PAIR(AMEDIA_IMGREADER_CANNOT_LOCK_IMAGE),
        MAKE_PAIR(AMEDIA_IMGREADER_CANNOT_UNLOCK_IMAGE),
        MAKE_PAIR(AMEDIA_IMGREADER_IMAGE_NOT_LOCKED),
};

const char *GetErrorStr(media_status_t err) {
    return GetPairStr<media_status_t>(err, errorInfo);
}