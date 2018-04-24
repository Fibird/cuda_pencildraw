find_package(OpenCV REQUIRED)

set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin/lib)

set(CPU_MATLAB_PATH /home/liuchaoyang/grad_proj/cuda_pracs/matlab_proj)
set(CPU_MATLAB_INCLUDE_DIRS ${CPU_MATLAB_PATH}/src ${CPU_MATLAB_PATH}/src/cpu_matlab)

include_directories(
${OpenCV_INCLUDE_DIRS}
${CPU_MATLAB_INCLUDE_DIRS}
)

add_library(GenPencil SHARED GenPencil.cpp)

link_directories(${OpenCV_LIBRARY_DIRS})

set(CPU_MATLAB_LIBS ${CPU_MATLAB_PATH}/bin/lib/libcpu_matlab.so)
target_link_libraries(GenPencil ${OpenCV_LIBS} ${CPU_MATLAB_LIBS})
