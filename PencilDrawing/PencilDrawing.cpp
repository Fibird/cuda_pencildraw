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
	int ks = 10;
	const int dir_num = 8;
	int data_type = CV_32FC1;

	if (argc != 2)
	{
		cout << "No arguments are specified!" << endl;
		return -1;
	}

	image = imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);
	genStroke(image, rst, 10, 1, 0.1);
	//image.convertTo(image, CV_32F, 1.0 / 255.0);
	//// Get the depth of input image
	//int img_dpt = image.depth();
	//// Get the size info of input image
	//unsigned int img_h, img_w;
	//img_h = image.rows;
	//img_w = image.cols;
	/*Size img_s = image.size();*/
	//
	//// Remove noise
	//medianBlur(image, image, 3);
	//
	/////// Get Image gradient
	//Mat grad_x, grad_y, grad;
	//grad_x = Mat::zeros(Size(img_w, img_h), CV_32FC1);
	//grad_y = Mat::zeros(Size(img_w, img_h), CV_32FC1);
	//// Gradient X
	//grad_x(Rect(1, 0, img_w - 1, img_h)) =
	//	abs(image(Rect(0, 0, img_w - 1, img_h)) - 
	//		image(Rect(1, 0, img_w - 1, img_h)));
	//// Gradient Y
	//grad_y(Rect(0, 0, img_w, img_h - 1)) = 
	//	abs(image(Rect(0, 0, img_w, img_h - 1)) - 
	//		image(Rect(0, 1, img_w, img_h - 1)));
	////addWeighted(grad_x, 0.5, grad_y, 0.5, 0, grad);
	//grad = grad_x + grad_y;

	/////// Classification
	//Mat ker_ref = Mat::zeros(ks * 2 + 1, ks * 2 + 1, CV_32FC1);

	//// Convolution kernel with horizontal direction
	//ker_ref(Rect(0, ks, ks * 2 + 1, 1)) = Mat::ones(1, ks * 2 + 1, CV_32FC1);

	//// Group gradient magnitudes
	//Mat response[dir_num];
	//Mat ker_real, rot_mat;
	//int rp_dpt = CV_32F;
	//for (int i = 0; i < dir_num; i++)
	//{
	//	rot_mat = getRotationMatrix2D(Point2f((float)ks, (float)ks), 
	//		(float)i * 180.0 / (float)dir_num, 1.0);
	//	warpAffine(ker_ref, ker_real, rot_mat, ker_ref.size());
	//	filter2D(grad, response[i], rp_dpt, ker_real);
	//}

	//Mat indices = Mat::zeros(img_s, CV_8U);
	//// Select the indices of maximum value
	//for (unsigned int i = 0; i < img_h * img_w; i++)
	//{
	//	char max_index = 0;
	//	float max_value = response[0].at<float>(i);
	//	for (int j = 1; j < dir_num; j++)
	//	{	
	//		float cur_value = response[j].at<float>(i);
	//		if (max_value < cur_value)
	//		{
	//			max_index = j;
	//			max_value = cur_value;
	//		}
	//	}
	//	indices.at<char>(i) = max_index;
	//}
	//
	//Mat C[dir_num];
	//int c_data_type = CV_32FC1;

	//for (int i = 0; i < dir_num; i++)
	//{
	//	C[i] = Mat::zeros(img_s, c_data_type);
	//}
	//// Get the magnitude map Cs for all directions
	//for (int i = 0; i < dir_num; i++)
	//{
	//	for (unsigned int j = 0; j < img_h * img_w; j++)
	//	{
	//		if (indices.at<char>(j) == i)
	//		{
	//			C[i].at<float>(j) = grad.at<float>(j);
	//		}
	//	}
	//}
	//
	////// Line shaping
	//Mat Spn[dir_num];
	//int sp_dpt = CV_32F;

	//for (int i = 0; i < dir_num; i++)
	//{
	//	rot_mat = getRotationMatrix2D(Point2f((float)ks, (float)ks), 
	//		(float)i * 180.0 / (float)dir_num, 1.0);
	//	warpAffine(ker_ref, ker_real, rot_mat, ker_ref.size());
	//	filter2D(C[i], Spn[i], sp_dpt, ker_real);
	//}

	//// Sum the result
	//Mat Sp = Spn[0];
	//for (int i = 1; i < dir_num; i++)
	//{
	//	Sp += Spn[i];
	//}

	//Sp.convertTo(Sp, CV_32FC1, 0.1);
	//Mat S = 1 - Sp;
	
	imshow("display", rst);
	waitKey(0);

    return 0;
}