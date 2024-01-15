cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(saicp_example)

# 设置 C++ 标准为 C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

find_package(PCL 1.8 REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable(saicp_example main.cpp)

# 链接 PCL 库和可能需要的 stdc++fs 库
target_link_libraries(saicp_example ${PCL_LIBRARIES})

