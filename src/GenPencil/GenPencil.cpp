#include "GenPencil.h"
#include "sparse_matrix.h"
#include <iostream>

void genPencil(const cv::Mat &pencil_texture, const cv::Mat &tone_map, const cv::Mat &stroke, cv::Mat &rst)
{
    if (tone_map.size() != stroke.size())
    {
        std::cout << "Size of Tone and Stroke is not identical!" << std::endl;
        return;
    }
    //// OpenCV Operations ////
    unsigned height = tone_map.size().height;
    unsigned width = tone_map.size().width;
    // Get P
	cv::Mat P;
    pencil_texture.convertTo(P, CV_64FC1, 1 / 255.0);
	cv::resize(P, P, cv::Size(width, height)); 
    // Get J
    cv::Mat J;
    tone_map.convertTo(J, CV_64FC1, 1 / 255.0);
    // Get S
    cv::Mat S;
    stroke.convertTo(S, CV_64FC1);

    //// Matrix Operations ////
    // Get data from OpenCV Mat
    Matrix m_P;
    m_P.rows = height;   m_P.cols = width;
    m_P.data = (double*)P.data;

    Matrix m_J;
    m_J.rows = height;   m_J.cols = width;
    m_J.data = (double*)J.data;

    Matrix m_S;
    m_S.rows = height; m_S.cols = width;
    m_S.data = (double*)S.data;

    Matrix m_T, all_ones;
    all_ones.rows = m_J.rows;    all_ones.cols = m_J.cols;
    ones(all_ones);
    
    // Compute pencil texture according to tone map
    sub(all_ones, m_J, m_J);
    pow(m_P, m_J, m_T);    

    Matrix m_rst;
    // Combine stroke and pencil texture
    dot_mul(m_S, m_T, m_rst);
    
    rst = cv::Mat(m_rst.rows, m_rst.cols, CV_64FC1, (double*)m_rst.data);
}
