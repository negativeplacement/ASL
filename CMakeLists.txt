cmake_minimum_required(VERSION 3.15)
project(Project VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

include(FetchContent)

FetchContent_Declare(
    opencv
    GIT_REPOSITORY https://github.com/opencv/opencv.git
    GIT_TAG        4.9.0
)

set(BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(BUILD_OPENCV_PYTHON_BINDINGS OFF CACHE BOOL "" FORCE)
set(BUILD_OPENCV_JAVA OFF CACHE BOOL "" FORCE)
set(WITH_CUDA OFF CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(opencv)

add_executable(project project.cpp)
target_link_libraries(project PUBLIC
    opencv_core
    opencv_dnn
    opencv_ml
    opencv_imgproc
    opencv_highgui
    opencv_imgcodecs
)

target_include_directories(project PUBLIC 
    ${opencv_SOURCE_DIR}/include
    ${opencv_BINARY_DIR}
)
