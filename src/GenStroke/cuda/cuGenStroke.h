#ifndef CU_GENSTROKE_H
#define CU_GENSTROKE_H

#define MAX_SPACE 4096
#include <cuda_runtime.h>
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

// Signal/image element type
// Must be float or double
typedef float element;

// constant memory used to save covolution kernel
__constant__ element Mask[MAX_SPACE];

void cuGenStroke(const cv::Mat & src, cv::Mat & dst, int ks, float gamma_s);

#endif
