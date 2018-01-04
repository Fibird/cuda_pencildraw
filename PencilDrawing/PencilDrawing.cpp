// PencilDrawing.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>

using namespace std;
using namespace cv;

int main(int argc, char** argv)
{
	if (argc != 2)
	{
		cout << "No arguments are specified!" << endl;
		return -1;
	}

	Mat image;
	int ks = 3;
	const int dir_num = 8;
	image = imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);
	int img_dpt = image.depth();
	
	// Remove noise
	medianBlur(image, image, 7);

	///// Get Image gradient
	Mat grad_x, grad_y, grad;
	// Gradient X
	Sobel(image, grad_x, img_dpt, 1, 0, 3, 1, 0, BORDER_DEFAULT);
	convertScaleAbs(grad_x, grad_x);
	// Gradient Y
	Sobel(image, grad_y, img_dpt, 0, 1, 3, 1, 0, BORDER_DEFAULT);
	convertScaleAbs(grad_y, grad_y);

	addWeighted(grad_x, 0.5, grad_y, 0.5, 0, grad);

	///// Classification
	Mat ker_ref = Mat::zeros(ks * 2 + 1, ks * 2 + 1, CV_16S);

	// Convolution kernel with horizontal direction
	for (int i = 0; i < ks * 2 + 1; i++)
	{
		ker_ref.at<int>(ks + 1, i) = 1;
	}
	
	// Groups gradient magnitudes
	Mat response[dir_num];
	Mat ker_real, rot_mat;
	for (int i = 0; i < dir_num; i++)
	{
		rot_mat = getRotationMatrix2D();
		//ker_real = 
	}
	//namedWindow("median", WINDOW_AUTOSIZE);
	imshow("display", grad_x);
	waitKey(0);

    return 0;
}

