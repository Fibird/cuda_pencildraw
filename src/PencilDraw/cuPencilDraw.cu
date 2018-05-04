#include <iostream>
#include <ctime>
#include "cuGenStroke.h"
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
    image.convertTo(image, CV_32FC1);
    
    cu_genStroke(image, gray_result, 10, 0.1f);
    return 0;
}
