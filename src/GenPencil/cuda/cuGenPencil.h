#ifndef CU_GENPENCIL_H
#define CU_GENPENCIL_H

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

#include <opencv2/core/core.hpp>

typedef float element;

void cuGenPencil(cv::Mat &pencil_texture, cv::Mat &tone_map, cv::Mat &stroke, cv::Mat &result);

#endif
