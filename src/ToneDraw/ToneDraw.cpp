#include <cmath>
#include <iostream>
#include "ToneDraw.h"
#include "cpu_histogram.h"

#define OMEGA1 76.0f	// 42, 52
#define OMEGA2 22.0f	// 29, 37
#define OMEGA3 2.0f		// 29, 11

#define PI 3.1415926535

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
			OMEGA3 * 1 / sqrtf(2 * PI * delta_d) * std::exp(-(i - mu_d) * 
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
