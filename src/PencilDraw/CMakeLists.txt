cmake_minimum_required(VERSION 3.5)

find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)

set(MAIN_PROJECT_ROOT_DIR ${PROJECT_SOURCE_DIR}/../..)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${MAIN_PROJECT_ROOT_DIR}/bin)

set(PENCIL_DRAW_INCLUDE_DIRS ${MAIN_PROJECT_ROOT_DIR}/src ${MAIN_PROJECT_ROOT_DIR}/src/GenStroke ${MAIN_PROJECT_ROOT_DIR}/src/ToneDraw ${MAIN_PROJECT_ROOT_DIR}/src/GenPencil)
set(CU_PENCIL_DRAW_INCLUDE_DIRS ${MAIN_PROJECT_ROOT_DIR}/src/GenStroke/cuda ${MAIN_PROJECT_ROOT_DIR}/src/ToneDraw/cuda ${MAIN_PROJECT_ROOT_DIR}/src/GenPencil/cuda)

include_directories(
${OpenCV_INCLUDE_DIRS}
${PENCIL_DRAW_INCLUDE_DIRS}
${CU_PENCIL_DRAW_INCLUDE_DIRS}
)

add_executable(cpu_pencil_draw PencilDraw.cpp)

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_60,code=sm_60)
cuda_add_executable(gpu_pencil_draw cuPencilDraw.cu)

set(PENCIL_DRAW_LIBRARY_DIRS ${MAIN_PROJECT_ROOT_DIR}/bin/lib)

set(PENCIL_DRAW_LIBS ${PENCIL_DRAW_LIBRARY_DIRS}/libGenStroke.so ${PENCIL_DRAW_LIBRARY_DIRS}/libToneDraw.so ${PENCIL_DRAW_LIBRARY_DIRS}/libGenPencil.so)
set(CU_PENCIL_DRAW_LIBS ${PENCIL_DRAW_LIBRARY_DIRS}/libcuGenStroke.a ${PENCIL_DRAW_LIBRARY_DIRS}/libcuToneDraw.a ${PENCIL_DRAW_LIBRARY_DIRS}/libcuGenPencil.a)
set(GPU_HISTOGRAM_PATH /home/liuchaoyang/grad_proj/cuda_pracs/histogram_proj)
set(GPU_HISTOGRAM_LIBS ${GPU_HISTOGRAM_PATH}/bin/lib/gpu_histogram/libgpu_histogram.a)
target_link_libraries(cpu_pencil_draw ${OpenCV_LIBS} ${PENCIL_DRAW_LIBS})
target_link_libraries(gpu_pencil_draw ${CU_PENCIL_DRAW_LIBS} ${OpenCV_LIBS} ${GPU_HISTOGRAM_LIBS})
