find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)

include_directories(
${OpenCV_INCLUDE_DIRS}
)

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_60,code=sm_60)
cuda_add_library(cuGenPencil STATIC cuGenPencil.cu)

target_link_libraries(cuGenPencil ${OpenCV_LIBS})
