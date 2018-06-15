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
    double all_time;

    start = clock();
	// Stroke Generation 
	genStroke(image, S_rst, 10, 1, 0.1f);
    // Tone Map Generation
	genToneMap(image, J_rst);
    // Pencil Texture Generation
    genPencil(pencil, J_rst, S_rst, gray_result);
    gray_result.convertTo(gray_result, CV_8UC1, 255);
    stop = clock();
    
    all_time = (double) (stop - start) / CLOCKS_PER_SEC;

    // Combine results
    //cvtColor(gray_result, color_result, COLOR_GRAY2RGBA);
    bool success = imwrite("result/cpu_gray_rst.png", gray_result);
    //imwrite("result/color_rst.png", color_result);
    cout << "***************************" << endl;
    cout << "* Pencil Draw(CPU)" << endl;
    cout << "* Created by Liu Chaoyang" << endl;
    cout << "***************************" << endl;
    if (success)
        cout << "======== Pencil Sketch Generatation Succeed ========" << endl;
    
    cout << "Elapsed Time of All: " << all_time << " sec" << endl;

    return 0;
}
