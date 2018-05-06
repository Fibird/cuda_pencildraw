#include <iostream>
#include <cmath>
#include <cuda_runtime.h>
#include <opencv2/imgproc/imgproc.hpp>
#include "cuGenPencil.h"

__global__ void _genPencil(element *J, element *P, element *S, element *T, unsigned width, unsigned height)
{
    int tidx = threadIdx.x + blockDim.x * blockIdx.x;
    int tidy = threadIdx.y + blockDim.y * blockIdx.y;
    int x_offset = blockDim.x * gridDim.x;
    int y_offset = blockDim.y * gridDim.y;
    while (tidx < width && tidy < height)
    {
        element value = 0;
        int id = tidy * width + tidx;
        value = pow(P[id], (1 - J[id]));
        T[id] = value * S[id];
        tidx += x_offset;
        tidy += y_offset;
    }
}

void cuGenPencil(cv::Mat &pencil_texture, cv::Mat &tone_map, cv::Mat &stroke, cv::Mat &result)
{
    if (tone_map.size() != stroke.size())
    {
        std::cout << "Size of Tone and Stroke is not identical!" << std::endl;
        return;
    }
    
    int type;
   // if (sizeof(element) == 32)
        type = CV_32FC1;
    //else
     //   type = CV_64FC1;

    //// OpenCV Operations ////
    unsigned height = tone_map.size().height;
    unsigned width = tone_map.size().width;
    // Get P
	cv::Mat P;
    pencil_texture.convertTo(P, type, 1 / 255.0);
	resize(P, P, cv::Size(width, height)); 
    // Get J
    cv::Mat J;
    tone_map.convertTo(J, type, 1 / 255.0);
    // Get S
    cv::Mat S;
    stroke.convertTo(S, type, 1 / 255.0);

    element *devJ, *devS, *devP, *devRst;

    CHECK(cudaMalloc((void**)&devJ, sizeof(element) * width * height)); 
    CHECK(cudaMalloc((void**)&devS, sizeof(element) * width * height)); 
    CHECK(cudaMalloc((void**)&devP, sizeof(element) * width * height)); 

    CHECK(cudaMemcpy(devJ, (element*)J.data, sizeof(element) * width * height, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(devP, (element*)P.data, sizeof(element) * width * height, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(devS, (element*)S.data, sizeof(element) * width * height, cudaMemcpyHostToDevice));

    CHECK(cudaMalloc((void**)&devRst, sizeof(element) * width * height)); 
    int ts_per_dm = 32;
    dim3 block(ts_per_dm, ts_per_dm);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    _genPencil<<<grid, block>>>(devJ, devP, devS, devRst, width, height);
    CHECK(cudaDeviceSynchronize());

    element *hostRst = (element*)malloc(sizeof(element) * width * height);
    CHECK(cudaMemcpy(hostRst, devRst, sizeof(element) * width * height, cudaMemcpyDeviceToHost));

    result = cv::Mat(height, width, type, hostRst);
}
