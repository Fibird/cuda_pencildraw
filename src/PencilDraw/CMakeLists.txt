find_package(OpenCV REQUIRED)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)

set(PENCIL_DRAW_INCLUDE_DIRS ${PROJECT_SOURCE_DIR}/src ${PROJECT_SOURCE_DIR}/src/GenStroke ${PROJECT_SOURCE_DIR}/src/ToneDraw ${PROJECT_SOURCE_DIR}/src/GenPencil)

include_directories(
${OpenCV_INCLUDE_DIRS}
${PENCIL_DRAW_INCLUDE_DIRS}
)

add_executable(pencil_draw PencilDraw.cpp)

set(PENCIL_DRAW_LIBRARY_DIRS ${PROJECT_SOURCE_DIR}/bin/lib)
link_directories(
${OpenCV_LIBRARY_DIRS}
${PENCIL_DRAW_LIBRARY_DIRS}
)

set(PENCIL_DRAW_LIBS ${PROJECT_SOURCE_DIR}/bin/lib/libGenStroke.so ${PROJECT_SOURCE_DIR}/bin/lib/libToneDraw.so ${PROJECT_SOURCE_DIR}/bin/lib/libGenPencil.so)
target_link_libraries(pencil_draw ${OpenCV_LIBS} ${PENCIL_DRAW_LIBS})
