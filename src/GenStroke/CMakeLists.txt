find_package(OpenCV REQUIRED)

set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin/lib)

set(CONV2D_PATH /home/liuchaoyang/grad_proj/cuda_pracs/conv_proj)
set(CONV2D_INCLUDE_DIRS ${CONV2D_PATH}/src/convolution)
include_directories(${OpenCV_INCLUDE_DIRS} ${CONV2D_INCLUDE_DIRS})

add_library(GenStroke SHARED GenStroke.cpp)

set(CONV2D_LIBS ${CONV2D_PATH}/bin/lib/convolution/libcpu_convolution.so)
target_link_libraries(GenStroke ${OpenCV_LIBS} ${CONV2D_LIBS})
add_subdirectory(cuda)
