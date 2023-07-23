cmake_minimum_required (VERSION 3.20)  # 需要至少 3.8 版本以支持 CUDA
project (myproject)
include_directories ("${PROJECT_SOURCE_DIR}/third_party/eigen")

# 查找 CUDA
find_package(CUDA REQUIRED)

# 将 CUDA 源文件编译成库
cuda_add_library(MyCUDAFiles src/layer/GPU_conv_baseline.cu)

# 添加 src 子目录
add_subdirectory(src)

# 指定生成目标
add_executable(train ${PROJECT_SOURCE_DIR}/train.cc)  
add_executable(inference ${PROJECT_SOURCE_DIR}/inference.cc)

# 添加链接库
target_link_libraries(train MiniDNNLib MyCUDAFiles)  
target_link_libraries(inference MiniDNNLib MyCUDAFiles) 
