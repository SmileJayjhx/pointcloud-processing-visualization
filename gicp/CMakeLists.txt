cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(icp_example)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

find_package(PCL 1.8 REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable(icp_example main.cpp)
target_link_libraries(icp_example ${PCL_LIBRARIES})
