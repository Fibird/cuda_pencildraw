# build pencil draw libraries
add_subdirectory(GenStroke)
add_subdirectory(ToneDraw)
add_subdirectory(GenPencil)

# build pencil_draw
find_package(OpenCV REQUIRED)

#set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)

set(PENCIL_DRAW_INCLUDE_DIRS ${PROJECT_SOURCE_DIRS}/src ${PROJECT_SOURCE_DIRS}/src/GenStroke ${PROJECT_SOURCE_DIRS}/src/ToneDraw ${PROJECT_SOURCE_DIRS}/src/GenPencil)
include_directories(
${OpenCV_INCLUDE_DIRS}
${PENCIL_DRAW_INCLUDE_DIRS}
)

add_executable(pencil_draw PencilDraw.cpp)

set(PENCIL_DRAW_LIBRARY_DIRS ${PROJECT_SOURCE_DIRS}/bin/lib)
link_directories(
${OpenCV_LIBRARY_DIRS}
${PENCIL_DRAW_LIBRARY_DIRS}
)

set(PENCIL_DRAW_LIBS libGenStroke.so libToneDraw.so libGenPencil.so)
target_link_libraries(pencil_draw ${OpenCV_LIBS} ${PENCIL_DRAW_LIBS})
