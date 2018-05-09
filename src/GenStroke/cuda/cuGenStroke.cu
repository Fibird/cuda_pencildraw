#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <cmath>
#include "cuGenStroke.h"
#include <iostream>

__global__ void getGrad(element *src, element *dst, unsigned width, unsigned height)
{
    int tidx = threadIdx.x + blockDim.x * blockIdx.x;
    int tidy = threadIdx.y + blockDim.y * blockIdx.y; 
    element grad_x = 0, grad_y = 0;

    if (tidx + 1 < width)
        grad_x = src[tidy * width + tidx] - src[tidy * width + tidx + 1];     
    if (tidy + 1 < height)
        grad_y = src[tidy * width + tidx] - src[(tidy + 1) * width + tidx];

    dst[tidy * width + tidx] = abs(grad_x) + abs(grad_y); 
}

__global__ void addLayers(element *arrays, element *result, unsigned width, unsigned height, int depth, float gamma_s)
{
    element value = 0;
    int tidx = threadIdx.x + blockDim.x * blockIdx.x;
    int tidy = threadIdx.y + blockDim.y * blockIdx.y;
   
    int x = tidx;   int y = tidy;
    int x_offset = blockDim.x * gridDim.x;
    int y_offset = blockDim.y * gridDim.y;

    while (x < width && y < height)
    {
        int idx;
        for (int i = 0; i < depth; i++)
        {
            idx = width * height * i + y * width + x;
            value += arrays[idx];
        }
        result[y * width + x] = 1 - value * gamma_s / 255.0;
        value = 0;
        x += x_offset;
        y += y_offset;
    }
}

__global__ void conv2D(const element *signal, element *result, unsigned width, unsigned height, int ks, int ts_per_dm, int order)
{
    int radius = ks / 2;
    // use dynamic size shared memory
    extern __shared__ element cache[];
    int sh_cols = ts_per_dm + radius * 2;
    int bk_cols = ts_per_dm;    int bk_rows = ts_per_dm;

	int gl_ix = threadIdx.x + blockDim.x * blockIdx.x;
	int gl_iy = threadIdx.y + blockDim.y * blockIdx.y;
    int ll_ix = threadIdx.x + radius;
    int ll_iy = threadIdx.y + radius;

	// Reads input elements into shared memory
	cache[ll_iy * sh_cols + ll_ix] = signal[gl_iy * width + gl_ix];
    // Marginal elements in cache
	if (threadIdx.x < radius)
	{
        int id = gl_iy * width + gl_ix - radius;
        cache[ll_iy * sh_cols + ll_ix - radius] = gl_ix < radius ? 0 : signal[id];
        id = gl_iy * width + gl_ix + bk_cols;
        cache[ll_iy * sh_cols + ll_ix + bk_cols] = gl_ix + bk_cols >= width ? 0 : signal[id];
	}
	if (threadIdx.y < radius)
	{
        int id = (gl_iy - radius) * width + gl_ix; 
        cache[(ll_iy - radius) * sh_cols + ll_ix] = gl_iy < radius ? 0 : signal[id];
        id = (gl_iy + bk_rows) * width + gl_ix;
        cache[(ll_iy + bk_rows) * sh_cols + ll_ix] = gl_iy + bk_rows >= height ? 0 :signal[id];
	}
    if (threadIdx.x < radius && threadIdx.y < radius)
    {
        int id = (gl_iy - radius) * width + gl_ix - radius;
        cache[(ll_iy - radius) * sh_cols + ll_ix - radius] = gl_iy < radius ? 0 : signal[id];
        id = (gl_iy - radius) * width + gl_ix + bk_cols;
        cache[(ll_iy - radius) * sh_cols + ll_ix + bk_cols] = gl_iy < radius ? 0 : signal[id];
        id = (gl_iy + bk_rows) * width + gl_ix + bk_cols;
        cache[(ll_iy + bk_rows) * sh_cols + ll_ix + bk_cols] = gl_iy + bk_rows >= height ? : signal[id];
        id = (gl_iy + bk_rows) * width + gl_ix - radius;
        cache[(ll_iy + bk_rows) * sh_cols + ll_ix - radius] = gl_iy + bk_rows >= height ? 0 : signal[id];
    }
	__syncthreads();

    // Get kernel element 
    element value = 0;
    for (int i = 0; i < ks; ++i)
	    for (int j = 0; j < ks; ++j)
	    	value  += cache[(ll_iy - radius + i) * sh_cols + ll_ix - radius + j] * Mask[order * ks * ks + i * ks + j];

	// Gets result 
    result[gl_iy * width + gl_ix] = value;
}

__global__ void getMagMap(element *resps, element *grad, element *cs, unsigned width, unsigned height, int order, int depth)
{
    int max_index = 0;
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int idy = threadIdx.y + blockDim.y * blockIdx.y;
    int x_offset = blockDim.x * gridDim.x;
    int y_offset = blockDim.y * gridDim.y;

    while (idx < width && idy < height)
    {
        element max_value = resps[width * height + idy * width + idx];
        for (int i = 1; i < depth; ++i)
        {
            element cur_value = resps[width * height * i + idy * width + idx];
            if (max_value < cur_value)
            {
                max_index = i;
                max_value = cur_value;
            }
        }
        cs[width * height * order + idy * width + idx] = (max_index == order) ? grad[idy * width + idx] : 0;
        idx += x_offset;
        idy += y_offset;
    }
}

__global__ void cu_medianfilter2DNoWrap(const element* signal, element* result, unsigned width, unsigned height, int k_width, int ts_per_dm)
{
	//element *kernel = (element*)malloc(sizeof(element) * k_width * k_width);
    element kernel[9];
    int radius = k_width / 2;
    // use dynamic size shared memory
    extern __shared__ element cache[];
    int sh_cols = ts_per_dm + radius * 2;
    int bk_cols = ts_per_dm;    int bk_rows = ts_per_dm;

	int gl_ix = threadIdx.x + blockDim.x * blockIdx.x;
	int gl_iy = threadIdx.y + blockDim.y * blockIdx.y;
    int ll_ix = threadIdx.x + radius;
    int ll_iy = threadIdx.y + radius;

	// Reads input elements into shared memory
	cache[ll_iy * sh_cols + ll_ix] = signal[gl_iy * width + gl_ix];
    // Marginal elements in cache
	if (threadIdx.x < radius)
	{
        int id = gl_iy * width + gl_ix - radius;
        cache[ll_iy * sh_cols + ll_ix - radius] = gl_ix < radius ? 0 : signal[id];
        id = gl_iy * width + gl_ix + bk_cols;
        cache[ll_iy * sh_cols + ll_ix + bk_cols] = gl_ix + bk_cols >= width ? 0 : signal[id];
	}
	if (threadIdx.y < radius)
	{
        int id = (gl_iy - radius) * width + gl_ix; 
        cache[(ll_iy - radius) * sh_cols + ll_ix] = gl_iy < radius ? 0 : signal[id];
        id = (gl_iy + bk_rows) * width + gl_ix;
        cache[(ll_iy + bk_rows) * sh_cols + ll_ix] = gl_iy + bk_rows >= height ? 0 :signal[id];
	}
    if (threadIdx.x < radius && threadIdx.y < radius)
    {
        int id = (gl_iy - radius) * width + gl_ix - radius;
        cache[(ll_iy - radius) * sh_cols + ll_ix - radius] = gl_iy < radius ? 0 : signal[id];
        id = (gl_iy - radius) * width + gl_ix + bk_cols;
        cache[(ll_iy - radius) * sh_cols + ll_ix + bk_cols] = gl_iy < radius ? 0 : signal[id];
        id = (gl_iy + bk_rows) * width + gl_ix + bk_cols;
        cache[(ll_iy + bk_rows) * sh_cols + ll_ix + bk_cols] = gl_iy + bk_rows >= height ? : signal[id];
        id = (gl_iy + bk_rows) * width + gl_ix - radius;
        cache[(ll_iy + bk_rows) * sh_cols + ll_ix - radius] = gl_iy + bk_rows >= height ? 0 : signal[id];
    }
	__syncthreads();

    // Get kernel element 
    for (int i = 0; i < k_width; ++i)
	    for (int j = 0; j < k_width; ++j)
	        kernel[i * k_width + j] = cache[(ll_iy - radius + i) * sh_cols + ll_ix - radius + j];

	// Orders elements (only half of them)
	for (int j = 0; j < k_width * k_width / 2 + 1; ++j)
	{
		// Finds position of minimum element
		int min = j;
		for (int k = j + 1; k < k_width * k_width; ++k)
			if (kernel[k] < kernel[min])
				min = k;
		// Puts found minimum element in its place
		const element temp = kernel[j];
		kernel[j] = kernel[min];
		kernel[min] = temp;
	}
	// Gets result - the middle element
	result[gl_iy * width + gl_ix] = kernel[k_width * k_width / 2];
}

__global__ void cu_medianfilter2D(const element* signal, element* result, unsigned width, unsigned height, int k_width, int ts_per_dm)
{
	//element *kernel = (element*)malloc(sizeof(element) * k_width * k_width);
    element kernel[9];
    int radius = k_width / 2;
    // use dynamic size shared memory
    extern __shared__ element cache[];
    int sh_cols = ts_per_dm + radius * 2;
    int bk_cols = ts_per_dm;    int bk_rows = ts_per_dm;
    unsigned sg_cols = width + radius * 2;

	int gl_ix = threadIdx.x + blockDim.x * blockIdx.x;
	int gl_iy = threadIdx.y + blockDim.y * blockIdx.y;
    int ll_ix = threadIdx.x + radius;
    int ll_iy = threadIdx.y + radius;

    if (gl_ix < width && gl_iy < height)
    {
	// Reads input elements into shared memory
	cache[ll_iy * sh_cols + ll_ix] = signal[(gl_iy + radius) * sg_cols + gl_ix + radius];
    // Marginal elements in cache
	if (threadIdx.x < radius)
	{
        cache[ll_iy * sh_cols + ll_ix - radius] = signal[(gl_iy + radius) * sg_cols + gl_ix];
        cache[ll_iy * sh_cols + ll_ix + bk_cols] = signal[(gl_iy + radius) * sg_cols + gl_ix + radius + bk_cols];
	}
	if (threadIdx.y < radius)
	{
        cache[(ll_iy - radius) * sh_cols + ll_ix] = signal[gl_iy * sg_cols + gl_ix + radius];
        cache[(ll_iy + bk_rows) * sh_cols + ll_ix] = signal[(gl_iy + radius + bk_rows) * sg_cols + gl_ix + radius];
	}
    if (threadIdx.x < radius && threadIdx.y < radius)
    {
         cache[(ll_iy - radius) * sh_cols + ll_ix - radius] = signal[gl_iy * sg_cols + gl_ix];
         cache[(ll_iy - radius) * sh_cols + ll_ix + bk_cols] = signal[gl_iy * sg_cols + gl_ix + radius + bk_cols];
         cache[(ll_iy + bk_rows) * sh_cols + ll_ix + bk_cols] = signal[(gl_iy + radius + bk_rows) * sg_cols + gl_ix + radius + bk_cols];
         cache[(ll_iy + bk_rows) * sh_cols + ll_ix - radius] = signal[(gl_iy + radius + bk_rows) * sg_cols + gl_ix];
    }
	__syncthreads();

    // Get kernel element 
    for (int i = 0; i < k_width; ++i)
	    for (int j = 0; j < k_width; ++j)
	    	kernel[i * k_width + j] = cache[(ll_iy - radius + i) * sh_cols + ll_ix - radius + j];

	// Orders elements (only half of them)
	for (int j = 0; j < k_width * k_width / 2 + 1; ++j)
	{
		// Finds position of minimum element
		int min = j;
		for (int k = j + 1; k < k_width * k_width; ++k)
			if (kernel[k] < kernel[min])
				min = k;
		// Puts found minimum element in its place
		const element temp = kernel[j];
		kernel[j] = kernel[min];
		kernel[min] = temp;
	}
	// Gets result - the middle element
	result[gl_iy * width + gl_ix] = kernel[k_width * k_width / 2];
    }
    //free(kernel);
}

void wrapImage(const element *src, element *dst, unsigned width, unsigned height, int ext_rad)
{
    if (ext_rad == 0)
        CHECK(cudaMemcpy(dst, (element*)src, width * height * sizeof(element), cudaMemcpyHostToHost));

	/////   Create image extension
    // Inner elements
    for (unsigned i = 0; i < height; ++i)
        CHECK(cudaMemcpy(dst + (width + ext_rad + ext_rad) * (i + ext_rad) +  ext_rad, (element*)src + width * i, width * sizeof(element), cudaMemcpyHostToHost));
    // marginal elements
    for (int i = 0; i < ext_rad; ++i)
    {
        CHECK(cudaMemcpy(dst + (width + ext_rad + ext_rad) * (ext_rad - i - 1), src + width * i, width * sizeof(element), cudaMemcpyHostToHost));
        CHECK(cudaMemcpy(dst + (width + ext_rad + ext_rad) * (height + ext_rad + i), src + width * (height - i - 1), width * sizeof(element), cudaMemcpyHostToHost)); 
    }
	for (int i = 0; i < height; ++i)
	{
        for (int j = 0; j < ext_rad; ++j)
        {
		    dst[(width + ext_rad + ext_rad) * (ext_rad + i) + j] = src[width * i + ext_rad - 1  - j];
		    dst[(width + ext_rad + ext_rad) * (ext_rad + i) + width + ext_rad + j] = src[width * i + width - (ext_rad - j - 1)];
        }
	}
}

void cuGenStroke(const cv::Mat &src, cv::Mat &dst, int kr, float gamma_s)
{
    // Get data from OpenCV
	cv::Mat img;
    int imgType;
    if (sizeof(element) == 4)
        imgType = CV_32FC1; 
    else
        imgType = CV_64FC1;

	src.convertTo(img, imgType, 1.0 / 255.0); // In most cases, the maximum of pixel value is 255
    unsigned height = src.size().height;
    unsigned width = src.size().width;

    int ts_per_dm = 32;
    dim3 med_block(ts_per_dm, ts_per_dm);
    dim3 med_grid((width + med_block.x - 1) / med_block.x, (height + med_block.y - 1) / med_block.y);
    
    ////--------------- medianfilter blur ----------------////
    int medKs = 3;
    int radius = medKs / 2;
    
    element *devSrc, *devMed;
	CHECK(cudaMalloc((void**)&devMed, width * height * sizeof(element)));
	CHECK(cudaMalloc((void**)&devSrc, width * height * sizeof(element)));

    cudaStream_t medStream;
    CHECK(cudaStreamCreate(&medStream));

    CHECK(cudaMemcpy(devSrc, (element*)src.data, width * height * sizeof(element), cudaMemcpyHostToDevice));

    unsigned med_shared_size = (ts_per_dm + 2 * radius) * (ts_per_dm + 2 * radius) * sizeof(element);
    
	cu_medianfilter2DNoWrap<<<med_grid, med_block, med_shared_size, medStream>>>(devSrc, devMed, width, height, medKs, ts_per_dm);

    // allocate data for grad 
    element *devGrad = devSrc;
	//CHECK(cudaMalloc((void**)&devGrad, width * height * sizeof(element)));
    
    CHECK(cudaStreamSynchronize(medStream));
    CHECK(cudaStreamDestroy(medStream));

    /////// medianfilter END

    //// ----------- Get gradient -------------- ////
    dim3 grad_block(ts_per_dm, ts_per_dm);
    dim3 grad_grid((width + grad_block.x - 1) / grad_block.x, (height + grad_block.y - 1) / grad_block.y);

    cudaStream_t gradStream;
    CHECK(cudaStreamCreate(&gradStream));

    getGrad<<<grad_grid, grad_block, 0, gradStream>>>(devMed, devGrad, width, height);

	const int dir_num = 8;
    element *devResps;
    // allocate data for conv2D
    CHECK(cudaMalloc((void**)&devResps, sizeof(element) * width * height * dir_num));

    CHECK(cudaStreamSynchronize(gradStream));
    CHECK(cudaStreamDestroy(gradStream));

    /////// gradient END

    //// ---------- convolution ----------- ////
    int ks = kr * 2 + 1;
	cv::Mat ker_ref = cv::Mat::zeros(ks, ks, CV_32FC1);
	ker_ref(cv::Rect(0, kr, ks, 1)) = cv::Mat::ones(1, ks, CV_32FC1);
    cv::Mat ker_real = cv::Mat::zeros(ks * 8, ks, CV_32FC1);
	cv::Mat rot_mat;
    // generate convolution kernels
    for (int i = 0; i < dir_num; i++)
    {
		rot_mat = getRotationMatrix2D(cv::Point2f((float)kr, (float)kr), (float)i * 180.0 / (float)dir_num, 1.0);
		// Get new kernel from ker_ref
		warpAffine(ker_ref, ker_real(cv::Rect(0, i * ks, ks, ks)), rot_mat, ker_ref.size());
    }
    CHECK(cudaMemcpyToSymbol(Mask, (element*)ker_real.data, sizeof(element) * ks * ks * dir_num));
    
    dim3 conv_block(ts_per_dm, ts_per_dm);
    dim3 conv_grid((width + conv_block.x - 1) / conv_block.x, (height + conv_block.y - 1) / conv_block.y);
    int conv_shared_size = (ts_per_dm + ks - 1) * (ts_per_dm + ks - 1) * sizeof(element);

    cudaStream_t conv2DStreams[dir_num];	
	for (int i = 0; i < dir_num; i++)
	{
        CHECK(cudaStreamCreate(&conv2DStreams[i]));
		// Convolution operation
        conv2D<<<conv_grid, conv_block, conv_shared_size, conv2DStreams[i]>>>(devGrad, devResps + width * height * i, width, height, ks, ts_per_dm, i);
	}
    
    // allocate memory for C
    element *devCs;
    CHECK(cudaMalloc((void**)&devCs, sizeof(element) * width * height * dir_num));

    for (int i = 0; i < dir_num; i++)
        CHECK(cudaStreamSynchronize(conv2DStreams[i]));

    //// ----------- Get magnitude map -------------- ////
    cudaStream_t magStreams[dir_num];
    dim3 mag_block(ts_per_dm, ts_per_dm);
    dim3 mag_grid((width + mag_block.x - 1) / mag_block.x, (height + mag_block.y - 1) / mag_block.y);
    for (int i = 0; i < dir_num; ++i)
    {
        CHECK(cudaStreamCreate(&magStreams[i]));
        getMagMap<<<mag_grid, mag_block, 0, magStreams[i]>>>(devResps, devGrad, devCs, width, height, i, dir_num);
    }

    // allocate memory for Spn
    element *devSpn = devResps;
    //CHECK(cudaMalloc((void**)&devSpn, sizeof(element) * width * height * dir_num));

    for (int i = 0; i < dir_num; i++)
    {
        CHECK(cudaStreamSynchronize(magStreams[i]));
        CHECK(cudaStreamDestroy(magStreams[i]));
    }

    //// ------------ convolution --------------- ////

    // Convolution operation
	for (int i = 0; i < dir_num; i++)
        conv2D<<<conv_grid, conv_block, conv_shared_size, conv2DStreams[i]>>>(devCs + width * height * i, devSpn + width * height * i, width, height, ks, ts_per_dm, i);
    element *devRst = devGrad;
    //CHECK(cudaMalloc((void**)&devRst, sizeof(element) * width * height)); 
    element *hostStrokeData;
    hostStrokeData = (element*)malloc(sizeof(element) * height * width);

    for (int i = 0; i < dir_num; i++)
    {
        CHECK(cudaStreamSynchronize(conv2DStreams[i]));
        CHECK(cudaStreamDestroy(conv2DStreams[i]));
    }

    //// ----------- combine ------------ ////
    dim3 add_block(ts_per_dm, ts_per_dm);
    dim3 add_grid((width + add_block.x - 1) / add_block.x, (height + add_block.y - 1) / add_block.y);
    addLayers<<<add_grid, add_block>>>(devSpn, devRst, width, height, dir_num, gamma_s);

    CHECK(cudaMemcpy(hostStrokeData, devRst, sizeof(element) * height * width, cudaMemcpyDeviceToHost));
    
    dst = cv::Mat(height, width, imgType, hostStrokeData);
    dst.convertTo(dst, CV_8UC1, 255.0);

    CHECK(cudaFree(devSrc));
    CHECK(cudaFree(devMed));
    CHECK(cudaFree(devResps));
    CHECK(cudaFree(devCs));
}
