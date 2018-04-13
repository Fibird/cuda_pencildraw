#include "stdafx.h"
#include "ToneDrawing.h"
#include <cmath>
#include <iostream>

#define OMEGA1 76.0f	// 42, 52
#define OMEGA2 22.0f	// 29, 37
#define OMEGA3 2.0f		// 29, 11

#define PI 3.1415926535

void hist_match(const cv::Mat &src, cv::Mat &dst, const double hgram[])
{
	if (!src.isContinuous())
	{
		std::cout << "The source image is not continuous!" << std::endl;
		exit(EXIT_FAILURE);
	}
	uchar *src_data = src.data;
	unsigned int rows = src.rows, cols = src.cols;

	// calculate histogram of source image
	int hist[256];
	memset(hist, 0, 256 * sizeof(int));
	for (unsigned int i = 0; i < rows * cols; ++i)
	{
		hist[src_data[i]]++;
	}
	// normalize the histogram
	double normal[256];
	unsigned int img_size = rows * cols;
	for (int i = 0; i < 256; ++i)
	{
		normal[i] = ((double)hist[i]) / img_size;
	}
	// compute cumulative histogram
	double src_cumulative[256], tgt_cumulative[256];
	double temp1 = 0.f, temp2 = 0.f;
	for (int i = 0; i < 256; ++i)
	{
		temp1 += normal[i];
		temp2 += hgram[i];
		src_cumulative[i] = temp1;
		tgt_cumulative[i] = temp2;
	}

	// using group map law(GML)
	int min_ids[256];
	int start = 0, end = 0, last_start = 0, last_end = 0;
	for (int i = 0; i < 256; ++i)
	{
		double min_value = abs(tgt_cumulative[i] - src_cumulative[0]);
		for (int j = 1; j < 256; ++j)
		{
			double temp = abs(tgt_cumulative[i] - src_cumulative[j]);
			if (temp <= min_value)
			{
				min_value = temp;
				end = j;
			}
		}
		if (start != last_start || end != last_end)
		{
			for (int t = start; t <= end; ++t)
			{
				// get relationship of mapping
				min_ids[t] = i;
			}
			last_start = start;
			last_end = end;
			start = last_end + 1;
		}
	}

	if (!dst.data)
	{
		dst.create(src.size(), src.type());
	}

	// map dst image according relationship
	uchar *dst_data = dst.data;
	for (unsigned int i = 0; i < rows * cols; ++i)
	{
		dst_data[i] = (uchar)min_ids[src_data[i]];
	}
}

void genToneMap(const cv::Mat & input, cv::Mat & J_rst)
{
	double target_histgram[256];
	float u_b = 225, u_a = 105;
	float mu_d = 90;
	float delta_b = 9, delta_d = 11;
	float total = 0;

	// Generating target histgram
	for (int i = 0; i < 256; i++)
	{
		target_histgram[i] = (
			// Model of bright layer
			OMEGA1 * (1 / delta_b) * std::exp(-(255 - i) / delta_b) +
			// Model of mild tone layer
			OMEGA2 * ((i >= u_a) && (i <= u_b) ? 1 / (u_b - u_a) : 0) + 
			// Model of dark layer
			OMEGA3 * 1 / std::sqrtf(2 * PI * delta_d) * std::exp(-(i - mu_d) * 
			(i - mu_d) / (2 * delta_d * delta_d))) * 0.01;
		total += target_histgram[i];
	}

	for (int i = 0; i < 256; i++)
	{
		target_histgram[i] /= total;
	}
	// process input from target histogram
	hist_match(input, J_rst, target_histgram);

	// average filter
	cv::blur(J_rst, J_rst, cv::Size(10, 10));
}
