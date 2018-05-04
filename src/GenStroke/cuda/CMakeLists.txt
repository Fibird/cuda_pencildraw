find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)

set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin/lib)

include_directories(${OpenCV_INCLUDE_DIRS})

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_60,code=sm_60)
cuda_add_library(cuGenStroke STATIC cuGenStroke.cu)

target_link_libraries(cuGenStroke ${OpenCV_LIBS} ${CUDA_LIBRARIES})
