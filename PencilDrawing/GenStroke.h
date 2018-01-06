#pragma once

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

// the input image should be grayscale image
void genStroke(const cv::Mat& src, cv::Mat& dst, int ks, int width, float gammaS);