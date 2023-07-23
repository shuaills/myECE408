cmake_minimum_required(VERSION 3.10)

# set the project name
project(MP0)

# specify the C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# find CUDA package
find_package(CUDA REQUIRED)

# set CUDA_NVCC_FLAGS
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_89,code=sm_89)

# include directories
include_directories(${CUDA_INCLUDE_DIRS})

# specify the executable
cuda_add_executable(MP0 mp0.cu)

# link libraries
target_link_libraries(MP0 ${CUDA_LIBRARIES})

add_library(wb STATIC IMPORTED)
set_property(TARGET wb PROPERTY IMPORTED_LOCATION ${CMAKE_SOURCE_DIR}/../libwb/lib/libwb.so)

target_link_libraries(MP0 wb)
