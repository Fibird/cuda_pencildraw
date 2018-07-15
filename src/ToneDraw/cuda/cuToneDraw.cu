#include <cuda_runtime.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <cmath>
#include "cuToneDraw.h"
#include "gpu_histogram.h"

#define OMEGA1 76.0f	// 42, 52
#define OMEGA2 22.0f	// 29, 37
#define OMEGA3 2.0f		// 29, 11

#define PI 3.1415926535

__global__ void genTargetHist(unsigned *tgtHist, unsigned *total)
{
    int blockNum = gridDim.x;
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int offset = blockDim.x * gridDim.x;
    *total = 0;

    double value = 0;
	double u_b = 225, u_a = 105;
	double mu_d = 90;
	double delta_b = 9, delta_d = 11;
    
    while (idx < 256)
    {
        value = ( 
			// Model of bright layer
			OMEGA1 * (1 / delta_b) * exp(-(255 - idx) / delta_b) +
			// Model of mild tone layer
			OMEGA2 * ((idx >= u_a) && (idx <= u_b) ? 1 / (u_b - u_a) : 0) + 
			// Model of dark layer
			OMEGA3 * 1 / sqrtf(2 * PI * delta_d) * exp(-(idx - mu_d) * 
			(idx - mu_d) / (2 * delta_d * delta_d))) * 0.01;
        value *= 100000; 
        tgtHist[idx] = value;
       // *total += value;
        atomicAdd(total, value);
        idx += offset;
    }
}

void cuGenToneMap(cv::Mat &input, cv::Mat &toneMap)
{
    if (!input.data)
        return;

    unsigned width, height;

    width = input.size().width;
    height = input.size().height;

    unsigned *devTgtHist;
    CHECK(cudaMalloc((void**)&devTgtHist, sizeof(element) * 256));

    unsigned *devTotal;
    CHECK(cudaMalloc((void**)&devTotal, sizeof(unsigned)));
    
    genTargetHist<<<16, 16>>>(devTgtHist, devTotal);
    CHECK(cudaDeviceSynchronize());

    unsigned total = 0;
    CHECK(cudaMemcpy(&total, devTotal, sizeof(unsigned), cudaMemcpyDeviceToHost));

    unsigned hostTgtHist[256];
    CHECK(cudaMemcpy(hostTgtHist, devTgtHist, sizeof(unsigned) * 256, cudaMemcpyDeviceToHost));

    cuHistMatch(input, toneMap, hostTgtHist, total);
	// average filter
	cv::blur(toneMap, toneMap, cv::Size(10, 10));
}
