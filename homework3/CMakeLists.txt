cmake_minimum_required(VERSION 3.10.0)
project(video_reader VERSION 0.1.0 LANGUAGES C CXX)

# 设置C++标准
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 查找OpenCV包
find_package(OpenCV REQUIRED)
find_package(Ceres REQUIRED)

# 包含头文件目录
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${CERES_INCLUDE_DIRS})

# 添加可执行文件
add_executable(video_reader src/main.cpp)

# 链接OpenCV与Ceres库
target_link_libraries(video_reader
    ${OpenCV_LIBS}
    ${CERES_LIBRARIES})

# 设置输出目录
set_target_properties(video_reader PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)