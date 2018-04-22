#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include "GenStroke.h"
#include "ToneDraw.h"

using namespace std;
using namespace cv;

int main(int argc, char** argv)
{
	Mat image, S_rst, J_rst;
	int data_type = CV_32FC1;

	if (argc != 2)
	{
		cout << "No arguments are specified!" << endl;
		return -1;
	}

	image = imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);
	// Stroke Generation 
	genStroke(image, S_rst, 10, 1, 0.1f);
	genToneMap(image, J_rst);
	imshow("S", S_rst);
	imshow("J", J_rst);
	waitKey(0);

    return 0;
}
