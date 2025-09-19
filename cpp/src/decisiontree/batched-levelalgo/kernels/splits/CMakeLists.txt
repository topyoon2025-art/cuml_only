cmake_minimum_required(VERSION 4.1)
project(example_applysplits LANGUAGES CXX CUDA)



set(CMAKE_PREFIX_PATH "/home/ubuntu/miniforge3/envs/rapids-25.08")
# Find RAFT and RMM
find_package(RAFT REQUIRED CONFIG)
find_package(RMM REQUIRED)
find_package(cuml REQUIRED CONFIG)


# Define your executable
add_executable(example_applysplits
  example_applysplits_main.cu
  example_ginisplits.cu
)

# Link against RAFT, RMM, cuML, and cuVS
target_link_libraries(example_applysplits PRIVATE
  raft::raft
  rmm::rmm
  cuml::cuml
  cuvs::cuvs
)

# Include RAFT and RMM headers
target_include_directories(example_applysplits PRIVATE
  ${RAFT_INCLUDE_DIRS}
  ${RMM_INCLUDE_DIRS}
)

# Set CUDA architecture
set_target_properties(example_applysplits PROPERTIES
  CUDA_ARCHITECTURES "80;89"
)
#include_directories(/home/ubuntu/cuml/cpp/src_prims)




