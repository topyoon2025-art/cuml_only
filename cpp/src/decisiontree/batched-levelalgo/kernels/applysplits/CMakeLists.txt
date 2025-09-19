#=============================================================================
# Copyright (c) 2019-2025, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#=============================================================================
cmake_minimum_required(VERSION 4.1)
project(example_applysplits LANGUAGES CXX CUDA)

add_executable(example_applysplits example_applysplits_main.cu example_apply.cu example_ginisplits.cu)
find_package(cuvs REQUIRED)
find_package(cuml REQUIRED)
target_link_libraries(example_applysplits PRIVATE cuml::cuml cuvs::cuvs)
include_directories(/home/ubuntu/miniforge3/envs/rapids-25.08/include)
include_directories(/home/ubuntu/cuml/cpp/src_prims)
set_target_properties(example_applysplits PROPERTIES
    CUDA_ARCHITECTURES "80;89"
)
#set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wno-deprecated-gpu-targets")




