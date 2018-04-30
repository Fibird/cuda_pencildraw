#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include "GenStroke.h"
#include "ToneDraw.h"
#include "GenPencil.h"
#include "matrix.h"

using namespace std;
using namespace cv;

int main(int argc, char** argv)
{
	Mat image, S_rst, J_rst;
    Mat pencil, T_rst;
	int data_type = CV_32FC1;

	if (argc != 3)
	{
		cout << "Usage: " << argv[0] << "input" << "pencil" << endl;
		return -1;
	}

	image = imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);
    pencil = imread(argv[2], CV_LOAD_IMAGE_GRAYSCALE);

	// Stroke Generation 
	genStroke(image, S_rst, 10, 1, 0.1f);
    // Tone Map Generation
	genToneMap(image, J_rst);
    // Pencil Texture Generation
    GenPencil(pencil, J_rst, T_rst, image.size().width, image.size().height);

    // Combine results
    S_rst.convertTo(S_rst, CV_64FC1);
    Matrix m_rst;
    Matrix m_S, m_T;
    m_S.rows = S_rst.size().height;    m_S.cols = S_rst.size().width;
    m_S.data = (double*)S_rst.data;
    m_T.rows = T_rst.size().height;    m_T.cols = T_rst.size().width;
    m_T.data = (double*)T_rst.data;
    dot_mul(m_S, m_T, m_rst);
    Mat rst(m_rst.rows, m_rst.cols, CV_64FC1, m_rst.data);
    rst.convertTo(rst, CV_8UC1, 255);
    imwrite("rst.png", rst);

    return 0;
}
