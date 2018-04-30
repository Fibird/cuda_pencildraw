cmake_minimum_required(VERSION 3.5)

find_package(OpenCV REQUIRED)

set(MAIN_PROJECT_ROOT_DIR ${PROJECT_SOURCE_DIR}/../..)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${MAIN_PROJECT_ROOT_DIR}/bin)

set(PENCIL_DRAW_INCLUDE_DIRS ${MAIN_PROJECT_ROOT_DIR}/src ${MAIN_PROJECT_ROOT_DIR}/src/GenStroke ${MAIN_PROJECT_ROOT_DIR}/src/ToneDraw ${MAIN_PROJECT_ROOT_DIR}/src/GenPencil)

include_directories(
${OpenCV_INCLUDE_DIRS}
${PENCIL_DRAW_INCLUDE_DIRS}
)

add_executable(pencil_draw PencilDraw.cpp)

set(PENCIL_DRAW_LIBRARY_DIRS ${MAIN_PROJECT_ROOT_DIR}/bin/lib)

set(PENCIL_DRAW_LIBS ${PENCIL_DRAW_LIBRARY_DIRS}/libGenStroke.so ${PENCIL_DRAW_LIBRARY_DIRS}/libToneDraw.so ${PENCIL_DRAW_LIBRARY_DIRS}/libGenPencil.so)
target_link_libraries(pencil_draw ${OpenCV_LIBS} ${PENCIL_DRAW_LIBS})
