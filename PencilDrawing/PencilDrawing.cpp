// PencilDrawing.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include "GenStroke.h"

using namespace std;
using namespace cv;

int main(int argc, char** argv)
{
	Mat image, rst;
	int data_type = CV_32FC1;

	if (argc != 2)
	{
		cout << "No arguments are specified!" << endl;
		return -1;
	}

	image = imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);
	// Stroke Generation 
	genStroke(image, rst, 10, 1, 0.1f);
	
	imshow("display", rst);
	waitKey(0);

    return 0;
}