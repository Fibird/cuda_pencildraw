find_package(OpenCV REQUIRED)

set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin/lib)

set(CPU_HISTOGRAM_PATH /home/liuchaoyang/grad_proj/cuda_pracs/histogram_proj)
set(CPU_HISTOGRAM_INCLUDE_DIRS ${CPU_HISTOGRAM_PATH}/src ${CPU_HISTOGRAM_PATH}/src/cpu_histogram)

include_directories(
${OpenCV_INCLUDE_DIRS}
${CPU_HISTOGRAM_INCLUDE_DIRS}
)

add_library(ToneDraw SHARED ToneDraw.cpp)

link_directories(${OpenCV_LIBRARY_DIRS})

set(CPU_HISTOGRAM_LIBS ${CPU_HISTOGRAM_PATH}/bin/lib/cpu_histogram/libcpu_histogram.so)
target_link_libraries(ToneDraw ${OpenCV_LIBS} ${CPU_HISTOGRAM_LIBS})
add_subdirectory(cuda)
