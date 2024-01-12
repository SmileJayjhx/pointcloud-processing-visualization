cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(SimulatedAnnealingICP)

# 设置 C++ 标准为 C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# 查找PCL库
find_package(PCL 1.8 REQUIRED)

# 包含PCL头文件
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

# 包含项目头文件
include_directories(include)

# 添加可执行文件
add_executable(gaicp src/saicp.cpp src/main.cpp)

# 链接PCL库到可执行文件
target_link_libraries(gaicp ${PCL_LIBRARIES})
