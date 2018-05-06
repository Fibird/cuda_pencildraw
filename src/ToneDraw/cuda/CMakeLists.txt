find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)

set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin/lib)

set(GPU_HISTOGRAM_PATH /home/liuchaoyang/grad_proj/cuda_pracs/histogram_proj)
set(GPU_HISTOGRAM_INCLUDE_DIRS ${GPU_HISTOGRAM_PATH}/src ${CPU_HISTOGRAM_PATH}/src/gpu_histogram)

include_directories(
${OpenCV_INCLUDE_DIRS}
${GPU_HISTOGRAM_INCLUDE_DIRS}
)

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_60,code=sm_60)
cuda_add_library(cuToneDraw STATIC cuToneDraw.cu)

set(GPU_HISTOGRAM_LIBS ${GPU_HISTOGRAM_PATH}/bin/lib/gpu_histogram/libgpu_histogram.a)
target_link_libraries(cuToneDraw ${OpenCV_LIBS} ${GPU_HISTOGRAM_LIBS})
