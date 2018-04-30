cmake_minimum_required(VERSION 3.5)

find_package(OpenCV REQUIRED)

set(MAIN_PROJECT_ROOT_DIR ${PROJECT_SOURCE_DIR}/../..)
message(${MAIN_PROJECT_ROOT_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${MAIN_PROJECT_ROOT_DIR}/bin)

set(PENCIL_DRAW_INCLUDE_DIRS ${MAIN_PROJECT_ROOT_DIR}/src ${MAIN_PROJECT_ROOT_DIR}/src/GenStroke ${MAIN_PROJECT_ROOT_DIR}/src/ToneDraw ${MAIN_PROJECT_ROOT_DIR}/src/GenPencil)

set(CPU_MATLAB_PATH /home/liuchaoyang/grad_proj/cuda_pracs/matlab_proj)
set(CPU_MATLAB_INCLUDE_DIRS ${CPU_MATLAB_PATH}/src ${CPU_MATLAB_PATH}/src/cpu_matlab)

include_directories(
${OpenCV_INCLUDE_DIRS}
${PENCIL_DRAW_INCLUDE_DIRS}
${CPU_MATLAB_INCLUDE_DIRS}
)

add_executable(pencil_draw PencilDraw.cpp)

set(PENCIL_DRAW_LIBRARY_DIRS ${MAIN_PROJECT_ROOT_DIR}/bin/lib)

set(CPU_MATLAB_LIBS ${CPU_MATLAB_PATH}/bin/lib/libcpu_matlab.so)

set(PENCIL_DRAW_LIBS ${PENCIL_DRAW_LIBRARY_DIRS}/libGenStroke.so ${PENCIL_DRAW_LIBRARY_DIRS}/libToneDraw.so ${PENCIL_DRAW_LIBRARY_DIRS}/libGenPencil.so)
target_link_libraries(pencil_draw ${OpenCV_LIBS} ${PENCIL_DRAW_LIBS} ${CPU_MATLAB_LIBS})
