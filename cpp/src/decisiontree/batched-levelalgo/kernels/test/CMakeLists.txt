cmake_minimum_required(VERSION 4.1)
project(test LANGUAGES CXX CUDA)

# Set CUDA architecture (adjust based on your GPU)
set(CMAKE_CUDA_ARCHITECTURES 80 89)

# Set your conda environment path
set(CONDA_PREFIX "/home/ubuntu/miniforge3/envs/rapids-25.08")
set(CMAKE_PREFIX_PATH "${CONDA_PREFIX}")

# Find packages
find_package(RAFT REQUIRED CONFIG)
find_package(RMM REQUIRED CONFIG)
find_package(cuml REQUIRED CONFIG)

# Define executable
add_executable(test main.cpp)

# Link libraries
target_link_libraries(test PRIVATE
  raft::raft
  rmm::rmm
  cuml::cuml
)

# Include directories manually if variables are not set
target_include_directories(test PRIVATE
  ${CONDA_PREFIX}/include
  ${CONDA_PREFIX}/include/raft
  ${CONDA_PREFIX}/include/rmm
  ${CONDA_PREFIX}/include/cuml
)

include_directories(/home/ubuntu/miniforge3/envs/rapids-25.08/include)

# Optional: If you have local source headers
#include_directories(/home/ubuntu/cuml/cpp/src_prims)
