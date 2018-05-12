#include <cmath>
#include <iostream>
#include "ToneDraw.h"
#include "cpu_histogram.h"

#define OMEGA1 76.0f	// 42, 52
#define OMEGA2 22.0f	// 29, 37
#define OMEGA3 2.0f		// 29, 11

#define PI 3.1415926535

void _avgfilter(uchar *src, uchar *dst, unsigned width, unsigned height, int ks)
{
    int radius = ks / 2;
    for (unsigned i = 0; i < height; ++i)
    {
        for (unsigned j = 0; j < width; ++j)
        {
            float value = 0;
            for (unsigned y = 0; y < ks; y++)
            {
                for (unsigned x = 0; x < ks; x++)
                {
                    unsigned ll_x = j - radius + x;
                    unsigned ll_y = i - radius + y;
                    if (!(ll_x < 0 || ll_x >= width || 
                        ll_y < 0 || ll_y >= height))
                        value += (float)src[ll_y * width + ll_x];
                }
            }
            dst[i * width + j] = value / (ks * ks);
        }
    }
}

void avgfilter(cv::Mat &src, cv::Mat &dst, int ks)
{
    unsigned width = src.size().width;
    unsigned height = src.size().height;

    if (!src.data)
        return;
    if (!src.isContinuous())
        return;

    dst = cv::Mat(src.size(), src.type());
    _avgfilter((uchar*)src.data, (uchar*)dst.data, width, height, ks);
}

void genToneMap(cv::Mat & input, cv::Mat & J_rst)
{
	int target_histgram[256];
	double u_b = 225, u_a = 105;
	double mu_d = 90;
	double delta_b = 9, delta_d = 11;
	unsigned total = 0;

	// Generating target histgram
	for (int i = 0; i < 256; i++)
	{
		target_histgram[i] = 100000 * (
			// Model of bright layer
			OMEGA1 * (1 / delta_b) * std::exp(-(255 - i) / delta_b) +
			// Model of mild tone layer
			OMEGA2 * ((i >= u_a) && (i <= u_b) ? 1 / (u_b - u_a) : 0) + 
			// Model of dark layer
			OMEGA3 * 1 / sqrtf(2 * PI * delta_d) * std::exp(-(i - mu_d) * 
			(i - mu_d) / (2 * delta_d * delta_d))) * 0.01;
		total += target_histgram[i];
	}

	// process input from target histogram
	hist_match(input, J_rst, target_histgram, total);

	// average filter
	//cv::blur(J_rst, J_rst, cv::Size(10, 10));
    cv::Mat temp;
    avgfilter(J_rst, temp, 9);
    J_rst = temp;
}
