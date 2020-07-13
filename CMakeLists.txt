cmake_minimum_required(VERSION 2.8.4)
project(Model_Deployment)
find_package(OpenCV REQUIRED)
find_package(dlib REQUIRED)

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O2")

include_directories(src)

add_library(Model_Deployment src/inference_framework.cpp)
target_link_libraries(Model_Deployment ${OpenCV_LIBRARIES} dlib::dlib)

add_executable(demo src/main.cpp)


target_link_libraries(demo ${OpenCV_LIBRARIES} dlib::dlib Model_Deployment)
