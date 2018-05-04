#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <cmath>
#include "cuGenStroke.h"

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
    int idx = width * height * depth + tidy * width + tidx;
    
    for (int i = 0; i < depth; i++)
        value += arrays[idx];
    
    result[idx] = 1 - value * gamma_s / 255.0;
}

__global__ void conv2D(const element *signal, element *result, unsigned width, unsigned height, int ks, int ts_per_dm)
{
    int radius = ks / 2;
    // use dynamic size shared memory
    extern __shared__ element cache[];
    int sh_cols = ts_per_dm + radius * 2;
    int bk_cols = ts_per_dm;    int bk_rows = ts_per_dm;
    unsigned sg_cols = width + radius * 2;

	int gl_ix = threadIdx.x + blockDim.x * blockIdx.x;
	int gl_iy = threadIdx.y + blockDim.y * blockIdx.y;
    int ll_ix = threadIdx.x + radius;
    int ll_iy = threadIdx.y + radius;

	// Reads input elements into shared memory
	cache[ll_iy * sh_cols + ll_ix] = signal[gl_iy * sg_cols + gl_ix];
    // Marginal elements in cache
	if (threadIdx.x < radius)
	{
        int id = gl_iy * sg_cols + gl_ix - radius;
        cache[ll_iy * sh_cols + ll_ix - radius] = gl_ix < radius ? 0 : signal[id];
        id = gl_iy * sg_cols + gl_ix + bk_cols;
        cache[ll_iy * sh_cols + ll_ix + bk_cols] = gl_ix + bk_cols >= width ? 0 : signal[id];
	}
	if (threadIdx.y < radius)
	{
        int id = (gl_iy - radius) * sg_cols + gl_ix; 
        cache[(ll_iy - radius) * sh_cols + ll_ix] = gl_iy < radius ? 0 : signal[id];
        id = (gl_iy + bk_rows) * sg_cols + gl_ix;
        cache[(ll_iy + bk_rows) * sh_cols + ll_ix] = gl_iy + bk_rows >= height ? 0 :signal[id];
	}
    if (threadIdx.x < radius && threadIdx.y < radius)
    {
        int id = (gl_iy - radius) * sg_cols + gl_ix - radius;
        cache[(ll_iy - radius) * sh_cols + ll_ix - radius] = gl_iy < radius ? 0 : signal[id];
        id = (gl_iy - radius) * sg_cols + gl_ix + bk_cols;
        cache[(ll_iy - radius) * sh_cols + ll_ix + bk_cols] = gl_iy < radius ? 0 : signal[id];
        id = (gl_iy + bk_rows) * sg_cols + gl_ix + bk_cols;
        cache[(ll_iy + bk_rows) * sh_cols + ll_ix + bk_cols] = gl_iy + bk_rows >= height ? : signal[id];
        id = (gl_iy + bk_rows) * sg_cols + gl_ix - radius;
        cache[(ll_iy + bk_rows) * sh_cols + ll_ix - radius] = gl_iy + bk_rows >= height ? 0 : signal[id];
    }
	__syncthreads();

    // Get kernel element 
    element value = 0;
    for (int i = 0; i < ks; ++i)
	    for (int j = 0; j < ks; ++j)
	    	value  = cache[(ll_iy - radius + i) * sh_cols + ll_ix - radius + j] * Mask[i * ks + j];

	// Gets result 
    result[gl_iy * width + gl_ix] = value / (ks * ks);
}

__global__ void genMag(element *resps, element *grad, element *cs, unsigned width, unsigned height, int order, int depth)
{
    int max_index = 0;
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int idy = threadIdx.y + blockDim.y * blockIdx.y;

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

    cs[width * height * order + idy * width +idx] = (max_index == order) ? grad[idy * width + idx] : 0;
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

void cu_genStroke(const cv::Mat &src, cv::Mat &dst, int ks, float gamma_s)
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
    dim3 block(ts_per_dm, ts_per_dm);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    
    ////--------------- medianfilter blur ----------------////
    element *hostExt;
    element *devExt, *devMed;
    int medKs = 3;
    int radius = medKs / 2;
	/////   Allocate page-locked memory for image extension 
	CHECK(cudaMallocHost((void**)&hostExt, (width + 2 * radius) * (height + 2 * radius) * sizeof(element)));
    
	/////   Create image extension
    // Inner elements
/*    for (unsigned i = 0; i < height; ++i)
        CHECK(cudaMemcpy(hostExt + (width + radius + radius) * (i + radius) +  radius, (element*)src.data + width * i, width * sizeof(element), cudaMemcpyHostToHost));
    // marginal elements
    for (int i = 0; i < radius; ++i)
    {
        CHECK(cudaMemcpy(hostExt + (width + radius + radius) * (radius - i - 1), src.data + width * i, width * sizeof(element), cudaMemcpyHostToHost));
        CHECK(cudaMemcpy(hostExt + (width + radius + radius) * (height + radius + i), src.data + width * (height - i - 1), width * sizeof(element), cudaMemcpyHostToHost)); 
    }
	for (int i = 0; i < height; ++i)
	{
        for (int j = 0; j < radius; ++j)
        {
		    hostExt[(width + radius + radius) * (radius + i) + j] = src.data[width * i + radius - 1  - j];
		    hostExt[(width + radius + radius) * (radius + i) + width + radius + j] = src.data[width * i + width - (radius - j - 1)];
        }
	}
*/
    wrapImage((element*)src.data, hostExt, width, height, radius);
    // Allocate device memory
	CHECK(cudaMalloc((void**)&devExt, (width + 2 * radius) * (height + 2 * radius) * sizeof(element)));
	CHECK(cudaMalloc((void**)&devMed, width * height * sizeof(element)));

	// Copies extension to device
	CHECK(cudaMemcpy(devExt, hostExt, (width + 2 * radius) * (height + 2 * radius) * sizeof(element), cudaMemcpyHostToDevice));

    unsigned shared_size = (ts_per_dm + 2 * radius) * (ts_per_dm + 2 * radius) * sizeof(element);
    
	cu_medianfilter2D<<<grid, block, shared_size>>>(devExt, devMed, width, height, medKs, ts_per_dm);
    cudaDeviceSynchronize();

    /////// medianfilter END

    //// ----------- Get gradient -------------- ////
    element *devGrad;
	CHECK(cudaMalloc((void**)&devGrad, width * height * sizeof(element)));

    getGrad<<<grid, block>>>(devMed, devGrad, width, height);
    cudaDeviceSynchronize();
    /////// gradient END

    //// ---------- convolution ----------- ////
	const int dir_num = 8;
	cv::Mat ker_ref = cv::Mat::zeros(ks * 2 + 1, ks * 2 + 1, CV_32FC1);
	ker_ref(cv::Rect(0, ks, ks * 2 + 1, 1)) = cv::Mat::ones(1, ks * 2 + 1, CV_32FC1);
	cv::Mat response[dir_num], ker_real, rot_mat;
    element *devResps;
    CHECK(cudaMalloc((void**)&devResps, sizeof(element) * width * height * dir_num));
	
	for (int i = 0; i < dir_num; i++)
	{
        // TODO:use cuda stream
		rot_mat = getRotationMatrix2D(cv::Point2f((float)ks, (float)ks),
			(float)i * 180.0 / (float)dir_num, 1.0);
		// Get new kernel from ker_ref
		warpAffine(ker_ref, ker_real, rot_mat, ker_ref.size());
        CHECK(cudaMemcpyToSymbol(Mask, (element*)ker_real.data, sizeof(element) * ks * ks));
		// Convolution operation
        conv2D<<<grid, block, shared_size>>>(devGrad, devResps + widht * height * i, width, height, ks, ts_per_dm);
        cudaDeviceSynchronize();
	}

    //// ----------- Get magnitude map -------------- ////
    element *devCs;
    CHECK(cudaMalloc((void**)&devCs, sizeof(element) * width * height * dir_num));

    for (int i = 0; i < dir_num; ++i)
        genMag<<<grid, block>>>(devResps, devGrad, devCs, width, height, i, dir_num);

    //// ------------ convolution --------------- ////
    element *devSpn;
    CHECK(cudaMalloc((void**)&devSpn, sizeof(element) * width * height * dir_num));

	for (int i = 0; i < dir_num; i++)
	{
        // TODO:use cuda stream
		rot_mat = getRotationMatrix2D(cv::Point2f((float)ks, (float)ks),
			(float)i * 180.0 / (float)dir_num, 1.0);
		// Get new kernel from ker_ref
		warpAffine(ker_ref, ker_real, rot_mat, ker_ref.size());
        CHECK(cudaMemcpyToSymbol(Mask, (element*)ker_real.data, sizeof(element) * ks * ks));
		// Convolution operation
        conv2D<<<grid, block, shared_size>>>(devCs, devSpn + width * height * i, width, height, ks, ts_per_dm);
        cudaDeviceSynchronize();
	}

    //// ----------- combine ------------ ////
    element *devRst;
    CHECK(cudaMalloc((void**)&devRst, sizeof(element) * width * height));
    
    addLayers<<<grid, block>>>(devSpn, devRst, width, height, dir_num, gamma_s);

    element *hostStrokeData;
    hostStrokeData = (element*)malloc(sizeof(element) * height * width);
    cudaMemcpy(hostStrokeData, devRst, sizeof(element) * height * width, cudaMemcpyDeviceToHost);
    
    dst = cv::Mat(height, width, imgType, hostStrokeData);
    dst.convertTo(dst, CV_8UC1, 255.0);
    imwrite("stroke.png", dst);
}
