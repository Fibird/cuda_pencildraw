#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <cuda_runtime.h>
#include <cmath>
#include <iostream>

#define CHECK(call)                                                            \
{                                                                              \
    const cudaError_t error = call;                                            \
    if (error != cudaSuccess)                                                  \
    {                                                                          \
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);                 \
        fprintf(stderr, "code: %d, reason: %s\n", error,                       \
                cudaGetErrorString(error));                                    \
        exit(1);                                                               \
    }                                                                          \
}

typedef float element;

#define MAX_SPACE 4096
// constant memory used to save covolution kernel
__constant__ element Mask[MAX_SPACE];

#define OMEGA1 76.0f	// 42, 52
#define OMEGA2 22.0f	// 29, 37
#define OMEGA3 2.0f		// 29, 11

#define PI 3.1415926535

__global__ void genTargetHist(unsigned *tgtHist)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int offset = blockDim.x * gridDim.x;

    double value = 0;
	double u_b = 225, u_a = 105;
	double mu_d = 90;
	double delta_b = 9, delta_d = 11;
    
    while (idx < 256)
    {
        value = ( 
			// Model of bright layer
			OMEGA1 * (1 / delta_b) * exp(-(255 - idx) / delta_b) +
			// Model of mild tone layer
			OMEGA2 * ((idx >= u_a) && (idx <= u_b) ? 1 / (u_b - u_a) : 0) + 
			// Model of dark layer
			OMEGA3 * 1 / sqrtf(2 * PI * delta_d) * exp(-(idx - mu_d) * 
			(idx - mu_d) / (2 * delta_d * delta_d))) * 0.01;
        value *= 100000; 
        tgtHist[idx] = value;
       // *total += value;
        atomicAdd(&(tgtHist[256]), value);
        idx += offset;
    }
}

__global__ void cuGetHist(uchar *src, unsigned *hist, unsigned width, unsigned height)
{
    __shared__ unsigned temp[256];
    temp[threadIdx.x] = 0;
    __syncthreads();

    int tidx = threadIdx.x + blockDim.x * blockIdx.x;
    int offset = blockDim.x * gridDim.x;

    // compute histogram
    int id = tidx;
    while (id < width * height)
    {
        atomicAdd(&(temp[src[id]]), 1);
        id += offset;
    }

    __syncthreads();
    atomicAdd(&(hist[threadIdx.x]), temp[threadIdx.x]);
}

__global__ void cuTgtMap(uchar *src, uchar *dst, unsigned *hist, unsigned width, unsigned height)
{
    int tidx = threadIdx.x + blockDim.x * blockIdx.x;
    int offset = blockDim.x * gridDim.x;

    // histogram map
    int id = tidx; // clear id
    while (id < width * height)
    {
        dst[id] = (uchar)hist[src[id]];
        id += offset;
    }
}

__global__ void avgFilter(uchar *signal, uchar *result, unsigned width, unsigned height, int ks, int ts_per_dm)
{
    int radius = ks / 2;
    // use dynamic size shared memory
    extern __shared__ uchar avg_cache[];
    int sh_cols = ts_per_dm + radius * 2;
    int bk_cols = ts_per_dm;    int bk_rows = ts_per_dm;

	int gl_ix = threadIdx.x + blockDim.x * blockIdx.x;
	int gl_iy = threadIdx.y + blockDim.y * blockIdx.y;
    int ll_ix = threadIdx.x + radius;
    int ll_iy = threadIdx.y + radius;

	// Reads input elements into shared memory
	avg_cache[ll_iy * sh_cols + ll_ix] = signal[gl_iy * width + gl_ix];
    // Marginal elements in cache
	if (threadIdx.x < radius)
	{
        int id = gl_iy * width + gl_ix - radius;
        avg_cache[ll_iy * sh_cols + ll_ix - radius] = gl_ix < radius ? 0 : signal[id];
        id = gl_iy * width + gl_ix + bk_cols;
        avg_cache[ll_iy * sh_cols + ll_ix + bk_cols] = gl_ix + bk_cols >= width ? 0 : signal[id];
	}
	if (threadIdx.y < radius)
	{
        int id = (gl_iy - radius) * width + gl_ix; 
        avg_cache[(ll_iy - radius) * sh_cols + ll_ix] = gl_iy < radius ? 0 : signal[id];
        id = (gl_iy + bk_rows) * width + gl_ix;
        avg_cache[(ll_iy + bk_rows) * sh_cols + ll_ix] = gl_iy + bk_rows >= height ? 0 :signal[id];
	}
    if (threadIdx.x < radius && threadIdx.y < radius)
    {
        int id = (gl_iy - radius) * width + gl_ix - radius;
        avg_cache[(ll_iy - radius) * sh_cols + ll_ix - radius] = gl_iy < radius ? 0 : signal[id];
        id = (gl_iy - radius) * width + gl_ix + bk_cols;
        avg_cache[(ll_iy - radius) * sh_cols + ll_ix + bk_cols] = gl_iy < radius ? 0 : signal[id];
        id = (gl_iy + bk_rows) * width + gl_ix + bk_cols;
        avg_cache[(ll_iy + bk_rows) * sh_cols + ll_ix + bk_cols] = gl_iy + bk_rows >= height ? : signal[id];
        id = (gl_iy + bk_rows) * width + gl_ix - radius;
        avg_cache[(ll_iy + bk_rows) * sh_cols + ll_ix - radius] = gl_iy + bk_rows >= height ? 0 : signal[id];
    }
	__syncthreads();

    // Get kernel element 
    element value = 0;
    for (int i = 0; i < ks; ++i)
	    for (int j = 0; j < ks; ++j)
	    	value  += avg_cache[(ll_iy - radius + i) * sh_cols + ll_ix - radius + j];

	// Gets result 
    result[gl_iy * width + gl_ix] = value / (ks * ks);
}

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

__global__ void conv2D(element *signal, element *result, unsigned width, unsigned height, int ks, int ts_per_dm, int order)
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

__global__ void cu_medianfilter2DNoWrap(element* signal, element* result, unsigned width, unsigned height, int k_width, int ts_per_dm)
{
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
	cache[ll_iy * sh_cols + ll_ix] = (element)signal[gl_iy * width + gl_ix];
    // Marginal elements in cache
	if (threadIdx.x < radius)
	{
        int id = gl_iy * width + gl_ix - radius;
        cache[ll_iy * sh_cols + ll_ix - radius] = gl_ix < radius ? 0 : (element)signal[id];
        id = gl_iy * width + gl_ix + bk_cols;
        cache[ll_iy * sh_cols + ll_ix + bk_cols] = gl_ix + bk_cols >= width ? 0 : (element)signal[id];
	}
	if (threadIdx.y < radius)
	{
        int id = (gl_iy - radius) * width + gl_ix; 
        cache[(ll_iy - radius) * sh_cols + ll_ix] = gl_iy < radius ? 0 : (element)signal[id];
        id = (gl_iy + bk_rows) * width + gl_ix;
        cache[(ll_iy + bk_rows) * sh_cols + ll_ix] = gl_iy + bk_rows >= height ? 0 : (element)signal[id];
	}
    if (threadIdx.x < radius && threadIdx.y < radius)
    {
        int id = (gl_iy - radius) * width + gl_ix - radius;
        cache[(ll_iy - radius) * sh_cols + ll_ix - radius] = gl_iy < radius ? 0 : (element)signal[id];
        id = (gl_iy - radius) * width + gl_ix + bk_cols;
        cache[(ll_iy - radius) * sh_cols + ll_ix + bk_cols] = gl_iy < radius ? 0 : (element)signal[id];
        id = (gl_iy + bk_rows) * width + gl_ix + bk_cols;
        cache[(ll_iy + bk_rows) * sh_cols + ll_ix + bk_cols] = gl_iy + bk_rows >= height ? : (element)signal[id];
        id = (gl_iy + bk_rows) * width + gl_ix - radius;
        cache[(ll_iy + bk_rows) * sh_cols + ll_ix - radius] = gl_iy + bk_rows >= height ? 0 : (element)signal[id];
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

__global__ void _genPencil(uchar *J, uchar *P, element *S, uchar *T, unsigned width, unsigned height)
{
    int tidx = threadIdx.x + blockDim.x * blockIdx.x;
    int tidy = threadIdx.y + blockDim.y * blockIdx.y;
    int x_offset = blockDim.x * gridDim.x;
    int y_offset = blockDim.y * gridDim.y;

    while (tidx < width && tidy < height)
    {
        element value = 0;
        int id = tidy * width + tidx;
        value = pow((element)P[id] / 255.0, (1 - (element)J[id] / 255.0));
        T[id] = value * S[id] * 255.0;
        //T[id] = S[id] * 255.0;
        tidx += x_offset;
        tidy += y_offset;
    }
}

void cuPencilDraw(cv::Mat &src, cv::Mat &fsrc, cv::Mat &pencil, cv::Mat &dst, int kr, float gamma_s)
{
    // Get data from OpenCV
	cv::Mat img;
    int imgType;
    if (sizeof(element) == 4)
        imgType = CV_32FC1; 
    else
        imgType = CV_64FC1;

    unsigned height = src.size().height;
    unsigned width = src.size().width;
	resize(pencil, pencil, cv::Size(width, height)); 

    ////--------------- Tone Map ----------------////
    unsigned *devTgtHist;
    CHECK(cudaMalloc((void**)&devTgtHist, sizeof(unsigned) * 257));
    CHECK(cudaMemset(devTgtHist + 256, 0, sizeof(unsigned)));
    cudaStream_t gtStream;
    CHECK(cudaStreamCreate(&gtStream));
    
    genTargetHist<<<16, 16, 0, gtStream>>>(devTgtHist);
    
    //================= Memcpy: DtoH ====================
    unsigned *hostTgtHist, hostTotal;
    CHECK(cudaMallocHost((void**)&hostTgtHist, sizeof(unsigned) * 257));
    CHECK(cudaMemcpyAsync(hostTgtHist, devTgtHist, sizeof(unsigned) * 257, cudaMemcpyDeviceToHost, gtStream));
    //========================================================

    /////////////// Malloc Device Memory ////////////////
    uchar *devSrc;
	CHECK(cudaMalloc((void**)&devSrc, width * height * sizeof(uchar)));
    ////////////////////////////////////////////////////
    CHECK(cudaMemcpy(devSrc, (uchar*)src.data, width * height * sizeof(uchar), cudaMemcpyHostToDevice));

    ///////////// Malloc Device Memory ///////////////
    unsigned *devSrcHist;
    CHECK(cudaMalloc((void**)&devSrcHist, sizeof(unsigned) * 256));
    CHECK(cudaMemset(devSrcHist, 0, sizeof(unsigned) * 256));

    cudaStream_t tmStream;
    CHECK(cudaStreamCreate(&tmStream));
    cudaDeviceProp prop;
    CHECK(cudaGetDeviceProperties(&prop, 0));
    int blocks = prop.multiProcessorCount;
    cuGetHist<<<blocks * 2, 256, 0, tmStream>>>(devSrc, devSrcHist, width, height);

    //================= Memcpy: DtoH =======================
    unsigned *hostSrcHist;
    CHECK(cudaMallocHost((void**)&hostSrcHist, sizeof(unsigned) * 256));
    CHECK(cudaMemcpyAsync(hostSrcHist, devSrcHist, sizeof(unsigned) * 256, cudaMemcpyDeviceToHost, tmStream));
    //======================================================

    /////////////// Malloc Device Memory ////////////////
    uchar *devToneMap;
    CHECK(cudaMalloc((void**)&devToneMap, sizeof(uchar) * width * height));
    ////////////////////////////////////////////////////


    //||||||||||||||||||sync barrier||||||||||||
    CHECK(cudaStreamSynchronize(gtStream));
    //||||||||||||||||||||||||||||||||||||||||||

    double srcAccums[256], tgtAccums[256];
    hostTotal = hostTgtHist[256];
    srcAccums[0] = hostSrcHist[0] / (width * height);
    tgtAccums[0] = hostTgtHist[0] / hostTotal;
    for (int i = 1; i < 256; ++i)
    {
        srcAccums[i] = srcAccums[i - 1] + (double)hostSrcHist[i] / (width * height);
        tgtAccums[i] = tgtAccums[i - 1] + (double)hostTgtHist[i] / hostTotal; 
    }

    unsigned *hostMins = hostSrcHist;
    // using group map law(GML)
    int start = 0, end = 0, last_start = 0, last_end = 0;
    for (int i = 0; i < 256; ++i)
    {
        double min_value = abs(tgtAccums[i] - srcAccums[0]); 
        for (int j = 1; j < 256; ++j)
        {
            double temp = abs(tgtAccums[i] - srcAccums[j]); 
            if (temp <= min_value)
            {
                min_value = temp;
                end = j;
            }
        }
        if (start != last_start || end != last_end)
        {
            for (int t = start; t <= end; ++t)
            {
                // get relationship of mapping
                hostMins[t] = i;
            }
            last_start = start;
            last_end = end;
            start = last_end + 1;
        }
    } 
    
    unsigned *devMins = devTgtHist;
    CHECK(cudaMemcpyAsync(devMins, hostMins, sizeof(unsigned) * 256, cudaMemcpyHostToDevice, tmStream));
    cuTgtMap<<<blocks * 2, 256, 0, tmStream>>>(devSrc, devToneMap, devMins, width, height);

    int ts_per_dm = 32, avgKs = 9;
    unsigned avg_shared_size = (ts_per_dm + avgKs - 1) * (ts_per_dm + 2 * avgKs - 1) * sizeof(element);
    dim3 avg_block(ts_per_dm, ts_per_dm);
    dim3 avg_grid((width + avg_block.x - 1) / avg_block.x, (height + avg_block.y - 1) / avg_block.y);
    uchar *devAvgTmp;
    CHECK(cudaMalloc((void**)&devAvgTmp, sizeof(uchar) * width * height));
    avgFilter<<<avg_grid, avg_block, avg_shared_size, tmStream>>>(devToneMap, devAvgTmp, width, height, avgKs, ts_per_dm);
    ///// End toneMap 

    ////--------------- medianfilter blur ----------------////
    int medKs = 3;
    int radius = medKs / 2;
    dim3 med_block(ts_per_dm, ts_per_dm);
    dim3 med_grid((width + med_block.x - 1) / med_block.x, (height + med_block.y - 1) / med_block.y);
    
    element *devMed;
	CHECK(cudaMalloc((void**)&devMed, width * height * sizeof(element)));

    unsigned med_shared_size = (ts_per_dm + 2 * radius) * (ts_per_dm + 2 * radius) * sizeof(element);
    element *devFsrc; 
	CHECK(cudaMalloc((void**)&devFsrc, width * height * sizeof(element)));
    CHECK(cudaMemcpy(devFsrc, (element*)fsrc.data, width * height * sizeof(element), cudaMemcpyHostToDevice));
	cu_medianfilter2DNoWrap<<<med_grid, med_block, med_shared_size, gtStream>>>(devFsrc, devMed, width, height, medKs, ts_per_dm);

    // allocate data for grad 
    element *devGrad;
	CHECK(cudaMalloc((void**)&devGrad, width * height * sizeof(element)));
    
    //CHECK(cudaStreamSynchronize(gtStream));


    /////// medianfilter END

    //// ----------- Get gradient -------------- ////
    dim3 grad_block(ts_per_dm, ts_per_dm);
    dim3 grad_grid((width + grad_block.x - 1) / grad_block.x, (height + grad_block.y - 1) / grad_block.y);

    getGrad<<<grad_grid, grad_block, 0, gtStream>>>(devMed, devGrad, width, height);

	const int dir_num = 8;
    element *devResps;
    // allocate data for conv2D
    CHECK(cudaMalloc((void**)&devResps, sizeof(element) * width * height * dir_num));
    
    //||||||||||||||| sync barrier |||||||||||||||
    CHECK(cudaStreamSynchronize(gtStream));
    //||||||||||||||||||||||||||||||||||||||||||||

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

    //||||||||||||||| sync barriers |||||||||||||||
    for (int i = 0; i < dir_num; i++)
        CHECK(cudaStreamSynchronize(conv2DStreams[i]));

    //// ----------- Get magnitude map -------------- ////
    dim3 mag_block(ts_per_dm, ts_per_dm);
    dim3 mag_grid((width + mag_block.x - 1) / mag_block.x, (height + mag_block.y - 1) / mag_block.y);
    for (int i = 0; i < dir_num; ++i)
        getMagMap<<<mag_grid, mag_block, 0, conv2DStreams[i]>>>(devResps, devGrad, devCs, width, height, i, dir_num);

    //||||||||||||||| sync barriers |||||||||||||||
    for (int i = 0; i < dir_num; i++)
        CHECK(cudaStreamSynchronize(conv2DStreams[i]));

    //// ------------ convolution --------------- ////
    element *devSpn = devResps;
    // Convolution operation
	for (int i = 0; i < dir_num; i++)
        conv2D<<<conv_grid, conv_block, conv_shared_size, conv2DStreams[i]>>>(devCs + width * height * i, devSpn + width * height * i, width, height, ks, ts_per_dm, i);
    element *devStroke = devGrad;

    for (int i = 0; i < dir_num; i++)
    {
        CHECK(cudaStreamSynchronize(conv2DStreams[i]));
        CHECK(cudaStreamDestroy(conv2DStreams[i]));
    }


    //// ----------- combine ------------ ////
    dim3 add_block(ts_per_dm, ts_per_dm);
    dim3 add_grid((width + add_block.x - 1) / add_block.x, (height + add_block.y - 1) / add_block.y);
    addLayers<<<add_grid, add_block, 0, gtStream>>>(devSpn, devStroke, width, height, dir_num, gamma_s);

    // Malloc memory
    uchar *devPen = devToneMap;
    //CHECK(cudaMalloc((void**)&devPen, sizeof(uchar) * width * height)); 
    CHECK(cudaMemcpy(devPen, pencil.data, sizeof(uchar) * width * height, cudaMemcpyHostToDevice));
    uchar *devRst = devSrc;
    CHECK(cudaStreamSynchronize(tmStream));
     
    dim3 gp_block(ts_per_dm, ts_per_dm);
    dim3 gp_grid((width + gp_block.x - 1) / gp_block.x, (height + gp_block.y - 1) / gp_block.y);

    _genPencil<<<gp_grid, gp_block, 0, gtStream>>>(devAvgTmp, devPen, devStroke, devRst, width, height);

    uchar *hostRst = (uchar*)malloc(sizeof(uchar) * height * width);

    CHECK(cudaStreamSynchronize(gtStream));

    CHECK(cudaMemcpy(hostRst, devRst, sizeof(uchar) * height * width, cudaMemcpyDeviceToHost));
    
    dst = cv::Mat(height, width, CV_8UC1, hostRst);

    CHECK(cudaFree(devTgtHist));    
    CHECK(cudaFree(devSrc));
    CHECK(cudaFree(devSrcHist));
    CHECK(cudaFreeHost(hostSrcHist));
    CHECK(cudaFreeHost(hostTgtHist));
    CHECK(cudaFree(devToneMap));
    CHECK(cudaFree(devMed));
    CHECK(cudaFree(devGrad));
    CHECK(cudaFree(devResps));
    CHECK(cudaFree(devCs));
    CHECK(cudaFree(devAvgTmp));
}

int main(int argc, char** argv)
{
	if (argc != 3)
	{
	    std::cout << "Usage: " << argv[0] << "input" << "pencil" << std::endl;
		return -1;
	}

	cv::Mat image = cv::imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);
    cv::Mat pencil = cv::imread(argv[2], CV_LOAD_IMAGE_GRAYSCALE);
    cv::Mat S_rst, J_rst, gray_result, color_result;
    cv::Mat fimage;
    image.convertTo(fimage, CV_32FC1); 

    clock_t start, stop;
    double all_time;
    
    // warm up cuda runtime
    char *warmup;
    cudaMalloc((void**)&warmup, sizeof(char));

    start = clock();
    cuPencilDraw(image, fimage, pencil, gray_result, 10, 0.1);
    stop = clock();
    all_time = (double) (stop - start) / CLOCKS_PER_SEC;

    imwrite("result/gpu_gray_rst.png", gray_result);

    std::cout << "Elapsed Time of All: " << all_time << " sec" << std::endl;

    return 0;
}
