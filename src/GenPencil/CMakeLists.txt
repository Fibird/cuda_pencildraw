find_package(OpenCV REQUIRED)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin/lib)

include_directories(${OpenCV_INCLUDE_DIRS})

add_library(GenPencil SHARED GenPencil.cpp)

link_directories(${OpenCV_LIBRARY_DIRS})

target_link_libraries(GenPencil ${OpenCV_LIBS})
