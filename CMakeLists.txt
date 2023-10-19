cmake_minimum_required(VERSION 3.5)
project(mytest VERSION 0.1.0 LANGUAGES C CXX)

include(CTest)
enable_testing()

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(mytest main.cpp)

target_link_libraries(mytest ${OpenCV_LIBS})
set(WITH_ONNX ON)
set(OPENCV_DNN_ONNX ON)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
