#include "GenPencil.h"
#include "sparse_matrix.h"

void GenPencil(const cv::Mat & input, const cv::Mat & pencil_texture, const cv::Mat & tone_map, cv::Mat & T_rst)
{
	double theta = 0.2;
	int w = input.size().width, h = input.size().height;
	cv::Mat P, J, logP, logJ;
	cv::resize(pencil_texture, P, input.size());
	P.reshape(0, w * h);
	cv::log(P, logP);
	// TODO:create sparse matrix from logP


	cv::resize(tone_map, J, input.size());
	J.reshape(0, w * h);
	cv::log(J, logJ);

	// 
}
