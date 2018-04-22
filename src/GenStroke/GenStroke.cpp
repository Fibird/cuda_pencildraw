#include "GenStroke.h"

void genStroke(const cv::Mat & src, cv::Mat & dst, int ks, int width, float gamma_s)
{
	//// Pre-process the input image
	// Scale pixel value to [0, 1] and change data type to float from char
	cv::Mat img = src.clone();
	img.convertTo(img, CV_32F, 1.0 / 255.0); // In most cases, the maximum of pixel value is 255
	// Remove noise
	cv::medianBlur(img, img, 3);
	cv::Size img_size = img.size();
	
	//// Get image gradient
	cv::Mat grad_x = cv::Mat::zeros(img_size, CV_32FC1);
	cv::Mat grad_y = cv::Mat::zeros(img_size, CV_32FC1);
	// Gradient X
	grad_x(cv::Rect(1, 0, img_size.width - 1, img_size.height)) =
		abs(img(cv::Rect(0, 0, img_size.width - 1, img_size.height)) -
			img(cv::Rect(1, 0, img_size.width - 1, img_size.height)));
	// Gradient Y
	grad_y(cv::Rect(0, 0, img_size.width, img_size.height - 1)) =
		abs(img(cv::Rect(0, 0, img_size.width, img_size.height - 1)) -
			img(cv::Rect(0, 1, img_size.width, img_size.height - 1)));
	cv::Mat grad = grad_x + grad_y;

	//// Classification by grouping gradient magnitudes
	// Create a convolution kernel with horizontal direction
	cv::Mat ker_ref = cv::Mat::zeros(ks * 2 + 1, ks * 2 + 1, CV_32FC1);
	ker_ref(cv::Rect(0, ks, ks * 2 + 1, 1)) = cv::Mat::ones(1, ks * 2 + 1, CV_32FC1);

	// Get the response maps
	const int dir_num = 8;
	cv::Mat response[dir_num], ker_real, rot_mat;
	
	for (int i = 0; i < dir_num; i++)
	{
		rot_mat = cv::getRotationMatrix2D(cv::Point2f((float)ks, (float)ks),
			(float)i * 180.0 / (float)dir_num, 1.0);
		// Get new kernel from ker_ref
		warpAffine(ker_ref, ker_real, rot_mat, ker_ref.size());
		// Convolution operation
		filter2D(grad, response[i], CV_32F, ker_real);
	}

	cv::Mat indices = cv::Mat::zeros(img_size, CV_8U);
	// Select the indices of maximum value
	for (int i = 0; i < img_size.area(); i++)
	{
		char max_index = 0;
		float max_value = response[0].at<float>(i);
		for (int j = 1; j < dir_num; j++)
		{
			float cur_value = response[j].at<float>(i);
			if (max_value < cur_value)
			{
				max_index = j;
				max_value = cur_value;
			}
		}
		indices.at<char>(i) = max_index;
	}

	cv::Mat C[dir_num];

	for (int i = 0; i < dir_num; i++)
	{
		C[i] = cv::Mat::zeros(img_size, CV_32FC1);
	}
	// Get the magnitude map Cs for all directions
	for (int i = 0; i < dir_num; i++)
	{
		for (int j = 0; j < img_size.area(); j++)
		{
			if (indices.at<char>(j) == i)
			{
				C[i].at<float>(j) = grad.at<float>(j);
			}
		}
	}

	//// Line shaping
	cv::Mat Spn[dir_num];

	for (int i = 0; i < dir_num; i++)
	{
		rot_mat = cv::getRotationMatrix2D(cv::Point2f((float)ks, (float)ks),
			(float)i * 180.0 / (float)dir_num, 1.0);
		warpAffine(ker_ref, ker_real, rot_mat, ker_ref.size());
		filter2D(C[i], Spn[i], CV_32F, ker_real);
	}

	// Sum the result
	cv::Mat Sp = Spn[0];

	for (int i = 1; i < dir_num; i++)
	{
		Sp += Spn[i];
	}

	Sp.convertTo(Sp, CV_32FC1, gamma_s);
	dst = 1 - Sp;
}
