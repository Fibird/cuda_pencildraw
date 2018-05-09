#include <iostream>
#include <ctime>
#include "GenStroke.h"
#include "ToneDraw.h"
#include "GenPencil.h"

using namespace std;
using namespace cv;

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
    
    clock_t start, stop;
    double gs_time, gt_time, gp_time;
    double all_time;

    start = clock();
	// Stroke Generation 
	genStroke(image, S_rst, 10, 1, 0.1f);
    stop = clock();
    gs_time = (double) (stop - start) / CLOCKS_PER_SEC;

    start = clock();
    // Tone Map Generation
	genToneMap(image, J_rst);
    stop = clock();
    gt_time = (double) (stop - start) / CLOCKS_PER_SEC;

    start = clock();
    // Pencil Texture Generation
    genPencil(pencil, J_rst, S_rst, gray_result);
    stop = clock();
    gp_time = (double) (stop - start) / CLOCKS_PER_SEC;

    // Combine results
    gray_result.convertTo(gray_result, CV_8UC1, 255);
    cvtColor(gray_result, color_result, COLOR_GRAY2RGBA);
    imwrite("result/cpu_gray_rst.png", gray_result);
    imwrite("result/color_rst.png", color_result);
    
    cout << "Elapsed Time of Generating Stroke: " << gs_time << " sec" << endl;
    cout << "Elapsed Time of Generating Tone Map: " << gt_time << " sec" << endl;
    cout << "Elapsed Time of Generating Pencil: " << gp_time << " sec" << endl;

    cout << "Elapsed Time of All: " << gs_time + gt_time + gp_time << " sec" << endl;

    return 0;
}
