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
	int ks = 10;
	const int dir_num = 8;
	image = imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);

	// Get the depth of input image
	int img_dpt = image.depth();
	// Get the size info of input image
	unsigned int img_h, img_w;
	img_h = image.rows;
	img_w = image.cols;
	Size img_s = image.size();

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
	Mat ker_ref = Mat::zeros(ks * 2 + 1, ks * 2 + 1, CV_8U);

	// Convolution kernel with horizontal direction
	for (int i = 0; i < ks * 2 + 1; i++)
	{
		ker_ref.at<char>(ks, i) = 1;
	}
	
	// Group gradient magnitudes
	Mat response[dir_num];
	Mat ker_real, rot_mat;
	for (int i = 0; i < dir_num; i++)
	{
		rot_mat = getRotationMatrix2D(Point2f(ks, ks), (i - 1) * 180 / dir_num, 1.0);
		warpAffine(ker_ref, ker_real, rot_mat, ker_ref.size());
		filter2D(grad, response[i], grad.depth(), ker_real);
	}
	
	Mat indices = Mat::zeros(img_s, CV_8U);
	// Select the indices of maximum value
	for (unsigned int i = 0; i < img_h * img_w; i++)
	{
		char max_index = 0;
		int max_value = response[0].at<char>(i);
		for (int j = 1; j < dir_num; j++)
		{	
			int cur_value = response[j].at<char>(i);
			if (max_value < cur_value)
			{
				max_index = j;
				max_value = cur_value;
			}
		}
		indices.at<char>(i) = max_index;
	}

	Mat C[dir_num];

	for (int i = 0; i < dir_num; i++)
	{
		C[i] = Mat::zeros(img_s, img_dpt);
	}
	// Get the magnitude map Cs for all directions
	for (int i = 0; i < dir_num; i++)
	{
		for (unsigned int j = 0; j < img_h * img_w; j++)
		{
			if (indices.at<char>(j) == i)
			{
				C[i].at<char>(j) = grad.at<char>(j);
			}
		}
	}
	
	//// Line shaping
	Mat Spn[dir_num];
	for (int i = 0; i < dir_num; i++)
	{
		rot_mat = getRotationMatrix2D(Point2f(ks, ks), (i - 1) * 180 / dir_num, 1.0);
		warpAffine(ker_ref, ker_real, rot_mat, ker_ref.size());
		filter2D(C[i], Spn[i], C[i].depth(), ker_real);
	}

	// Sum the result
	Mat Sp = Spn[0];
	for (unsigned int i = 0; i < img_h * img_w; i++)
	{
		for (int j = 1; j < dir_num; j++)
		{
			Sp.at<char>(i) += Spn[j].at<char>(i);
		}
	}

	// Map to [0, 1]
	Mat S;
	Sp.convertTo(Sp, CV_32FC1);
	//normalize(Sp, S, 0.0, 1.0);
	//namedWindow("median", WINDOW_AUTOSIZE);
	S = 1 - Sp;
	imshow("display", S);
	waitKey(0);

    return 0;
}