#include <iostream>
#include <ctime>
#include "cuGenStroke.h"
#include "cuToneDraw.h"
#include "cuGenPencil.h"
#include <opencv2/highgui/highgui.hpp>

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
    Mat fImg;
    image.convertTo(fImg, CV_32FC1);
    
    cuGenStroke(fImg, S_rst, 10, 0.1f);
    cuGenToneMap(image, J_rst);
    cuGenPencil(pencil, J_rst, S_rst, gray_result);

    gray_result.convertTo(gray_result, CV_8UC1, 255.0);
    
    imwrite("result/gray_result.png", gray_result);

    return 0;
}
