#pragma once
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

void GenPencil(const cv::Mat &pencil_texture, const cv::Mat &tone_map, const cv::Mat &stroke, cv::Mat &rst);
