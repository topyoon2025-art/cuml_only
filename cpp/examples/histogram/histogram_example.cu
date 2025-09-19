#pragma once
#include <cuda_runtime.h>
#include <iostream>
#include "histogram_example.cuh"
#include "/home/ubuntu/cuml/cpp/src/decisiontree/batched-levelalgo/kernels/builder_kernels.cuh"
#include "/home/ubuntu/cuml/cpp/src/decisiontree/batched-levelalgo/bins.cuh"


__global__ void myKernel() {
    printf("Hello from GPU!\n");
}

void launchKernel() {
    myKernel<<<1, 1>>>();
    cudaDeviceSynchronize();
}

namespace ML {
namespace DT {

static constexpr int TPB_DEFAULT = 128;
template <typename DatasetT, typename NodeT, typename ObjectiveT, typename DataT>
static __global__ void leafKernel(ObjectiveT objective,
                                  DatasetT dataset,
                                  const NodeT* tree,
                                  const InstanceRange* instance_ranges,
                                  DataT* leaves)
{
  using BinT = typename ObjectiveT::BinT;
  extern __shared__ char shared_memory[];
  auto histogram = reinterpret_cast<BinT*>(shared_memory);
  auto node_id   = blockIdx.x;
  auto& node     = tree[node_id];
  auto range     = instance_ranges[node_id];
  if (!node.IsLeaf()) return;
  auto tid = threadIdx.x;
  for (int i = tid; i < dataset.num_outputs; i += blockDim.x) {
    histogram[i] = BinT();
  }
  __syncthreads();
  for (auto i = range.begin + tid; i < range.begin + range.count; i += blockDim.x) {
    auto label = dataset.labels[dataset.row_ids[i]];
    BinT::IncrementHistogram(histogram, 1, 0, label);
  }
  __syncthreads();
  if (tid == 0) {
    ObjectiveT::SetLeafVector(
      histogram, dataset.num_outputs, leaves + dataset.num_outputs * node_id);
  }
}
}  // namespace DT
}  // namespace ML