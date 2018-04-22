#pragma once

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

void genToneMap(const cv::Mat& input, cv::Mat& J_rst);
void hist_match(const cv::Mat &src, cv::Mat &dst, const double hgram[]);