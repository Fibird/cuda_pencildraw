#ifndef CU_GEN_TONE_MAP_H
#define CU_GEN_TONE_MAP_H

#include <opencv2/core/core.hpp>

#define CHECK(call)                                                            \
{                                                                              \
    const cudaError_t error = call;                                            \
    if (error != cudaSuccess)                                                  \
    {                                                                          \
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);                 \
        fprintf(stderr, "code: %d, reason: %s\n", error,                       \
                cudaGetErrorString(error));                                    \
        exit(1);                                                               \
    }                                                                          \
}

typedef float element;

void cuGenToneMap(cv::Mat &src, cv::Mat &dst);

#endif
