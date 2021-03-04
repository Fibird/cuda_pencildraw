#include <iostream>
#include <ctime>
#include <cuda_runtime.h>
#include "cuGenStroke.h"
#include "cuToneDraw.h"
#include "cuGenPencil.h"
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;

__global__ void warmup(char *w)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx == 0)
        printf("Warming up ...\n");
}

int main(int argc, char** argv)
{
	if (argc != 3)
	{
		cout << "Usage: " << argv[0] << "input" << "pencil" << endl;
		return -1;
	}

	Mat image = imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);
    Mat pencil = imread(argv[2], CV_LOAD_IMAGE_GRAYSCALE);
    Mat S_rst, J_rst, gray_result, color_result;
    Mat fImg;
    image.convertTo(fImg, CV_32FC1);
    
    clock_t start, stop;
    double all_time;
    
    cout << "***************************" << endl;
    cout << "* Pencil Draw(Stream) V1.2" << endl;
    cout << "* Created by Liu Chaoyang" << endl;
    cout << "***************************" << endl;
    printf("Initializing CUDA runtime environment...\n");
    // warm up cuda runtime
    char *warmup;
    cudaMalloc((void**)&warmup, sizeof(char));
    printf("Initialization Complete. Start running program.\n");

    start = clock();
    cuGenStroke(fImg, S_rst, 10, 0.1f);
    cuGenToneMap(image, J_rst);
    cuGenPencil(pencil, J_rst, S_rst, gray_result);
    stop = clock();
    all_time = (double) (stop - start) / CLOCKS_PER_SEC;
    //bool success = imwrite("result/stroke.png", S_rst);
 //   imwrite("result/tone.png", J_rst);

    gray_result.convertTo(gray_result, CV_8UC1, 255.0);
    
    bool success = imwrite("result/gpu_gray_rst.png", gray_result);
    if (success)
        cout << "======== Pencil Sketch Generatation Succeed ========" << endl;

    cout << "Elapsed Time of All: " << all_time << " sec" << endl;

    return 0;
}
