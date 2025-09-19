#include <cuda_runtime.h>
#include <iostream>
#include "../builder_kernels_impl.cuh" // includes computeSplitKernel and related structs
#include "../../dataset.h"  // Make sure this path is correct and that dataset.h defines Dataset
#include "../../objectives.cuh"
#include <unordered_set>
#include "utils.hpp"
#include <cfloat>

#define CUDA_CHECK(call)                                         \
  do {                                                           \
    cudaError_t err = call;                                      \
    if (err != cudaSuccess) {                                    \
      std::cerr << "CUDA error at " << __FILE__ << ":"          \
                << __LINE__ << ": " << cudaGetErrorString(err)  \
                << std::endl;                                   \
      std::exit(EXIT_FAILURE);                                  \
    }                                                            \
  } while (0)

__global__ void warmup1() {}

__global__ void ComputeMinMaxKernel (
    const float* __restrict__ d_col_add_projected,
    float* __restrict__ d_block_min,
    float* __restrict__ d_block_max,
    int num_rows,
    int num_proj
){
    extern __shared__ float shared[];
    float* shared_min = shared;
    float* shared_max = shared + blockDim.x;

    int proj_id = blockIdx.y;
    int tid     = threadIdx.x;
    if (proj_id >= num_proj) return;

    int base_idx = proj_id * num_rows;

    float local_min = FLT_MAX;
    float local_max = -FLT_MAX;

    // strided loop over rows
    for (int i = blockIdx.x * blockDim.x + tid; i < num_rows; i += gridDim.x * blockDim.x) {
        float val = d_col_add_projected[base_idx + i];
        local_min = fminf(local_min, val);
        local_max = fmaxf(local_max, val);
    }

    // store per-thread local min/max in shared
    shared_min[tid] = local_min;
    shared_max[tid] = local_max;
    __syncthreads();

    // block-wide reduction
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_min[tid] = fminf(shared_min[tid], shared_min[tid + stride]);
            shared_max[tid] = fmaxf(shared_max[tid], shared_max[tid + stride]);
        }
        __syncthreads();
    }

    // write block result
    if (tid == 0) {
        int idx = proj_id * gridDim.x + blockIdx.x;
        d_block_min[idx] = shared_min[0];
        d_block_max[idx] = shared_max[0];
    }
}


__global__ void FinalMinMaxKernel(
    const float* __restrict__ d_block_min,
    const float* __restrict__ d_block_max,
    float* __restrict__ d_min_vals,
    float* __restrict__ d_max_vals,
    float* __restrict__ d_bin_widths,
    int num_blocks,
    int num_proj,
    int num_bins
) {
    int proj_id = blockIdx.x;
    if (proj_id >= num_proj) return;

    float min_val = FLT_MAX;
    float max_val = -FLT_MAX;

    // sequential reduction across all blocks of this projection
    for (int i = 0; i < num_blocks; ++i) {
        int idx = proj_id * num_blocks + i;
        min_val = fminf(min_val, d_block_min[idx]);
        max_val = fmaxf(max_val, d_block_max[idx]);
    }

    d_min_vals[proj_id] = min_val;
    d_max_vals[proj_id] = max_val;
    d_bin_widths[proj_id] = (max_val > min_val)
                            ? (max_val - min_val) / (float)num_bins
                            : 1.0f;
}




template <typename DataT, typename LabelT, typename IdxT>
void GiniGainsBestSplits (DataT* d_projected,
                          const std::vector<LabelT>& h_labels,
                          const IdxT M,    // number of samples/rows
                          const IdxT N,       // total number of features, num_proj
                          const IdxT n_sampled_cols,       // Sample every feature
                          const IdxT num_nodes,       // two nodes in this example
                          const IdxT max_n_bins,
                          const IdxT max_depth,//not used
                          const IdxT min_samples_split,
                          const IdxT max_leaves,//not used
                          const uint64_t seed,
                          const IdxT treeid,
                          const IdxT colStart,
                          const IdxT num_classes, // number of classes for classification, binary for this example
                          const IdxT min_samples_leaf,
                          ML::DT::GiniObjectiveFunction<DataT, LabelT, IdxT>& objective,
                          std::vector<ML::DT::Split<DataT, IdxT>>& h_splits,
                          double& elapsed_ms,
                          bool verbose) {


//////////////////////////ComputeMinMax///////////////////////////////////////////////////////
    warmup1<<<1, 1>>>();
    cudaDeviceSynchronize();

    float* d_min_vals;
    float* d_max_vals;
    float* d_bin_widths;
    cudaMalloc(&d_min_vals, N * sizeof(float));
    cudaMalloc(&d_max_vals, N * sizeof(float));
    cudaMalloc(&d_bin_widths, N * sizeof(float));

    int threads_per_block_minmax = 256; //using 256 threads per block for min/max computation
    int blocks_per_projection = 256;

    // safety: must be >= 1
    blocks_per_projection = std::max(1, blocks_per_projection);
    const int total_blocks = N * blocks_per_projection;

     // Allocate intermediate buffers
    float* d_block_min;
    float* d_block_max;
    cudaMalloc(&d_block_min, total_blocks * sizeof(float));
    cudaMalloc(&d_block_max, total_blocks * sizeof(float));

    dim3 grid1(blocks_per_projection, N);
    dim3 block1(threads_per_block_minmax);
    size_t shmem = 2 * threads_per_block_minmax * sizeof(float);
    
    // Launch Pass 1: local min/max per block
    // std::cout << "Launching ComputeMinMaxKernel with grid=("
    //       << grid1.x << "," << grid1.y << "), block=("
    //       << block1.x << "), shared=" << shmem << " bytes\n";
    // std::cout << "num_proj = " << N << ", num_rows = " << M << "\n";
    auto start0 = std::chrono::high_resolution_clock::now();
    ComputeMinMaxKernel<<<grid1, block1, shmem>>>(d_projected,
                                                d_block_min,
                                                d_block_max,
                                                M,
                                                N);
    cudaError_t err0 = cudaGetLastError();
    cudaDeviceSynchronize();
    auto end0 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration0 = end0 - start0;
    printf("ComputeMinMaxKernel time elapsed: %f ms\n", duration0.count());

    // Launch Pass 2: final reduction per projection
    auto start1 = std::chrono::high_resolution_clock::now();
    FinalMinMaxKernel<<<N, 1>>>(
        d_block_min,
        d_block_max,
        d_min_vals,
        d_max_vals,
        d_bin_widths,
        blocks_per_projection,
        N,
        max_n_bins
    );
    cudaError_t err1 = cudaGetLastError();
    cudaDeviceSynchronize();
 
    auto end1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration1 = end1 - start1;
    std::cout << "Final Min/Max time elapsed: " << duration1.count() << " ms " << std::endl;

    if (err0 != cudaSuccess) {
        printf("Kernel launch failed min/max Phase I: %s\n", cudaGetErrorString(err0));
    }
    if (err1 != cudaSuccess) {
        printf("Kernel launch failed min/max Phase II: %s\n", cudaGetErrorString(err0));
    }

    // Cleanup
    cudaFree(d_block_min);
    cudaFree(d_block_max);

  //////////////////////////ComputeMinMax/////////////////////////////////////////////////////// 
    
  std::vector<float> h_min_vals(N); 
  CUDA_CHECK(cudaMemcpy(h_min_vals.data(), d_min_vals,
                          N * sizeof(float),
                          cudaMemcpyDeviceToHost));
  if (verbose) {
    printf("Min val: ");
    for (int i = 0; i < N; ++i) {
        printf("%f, ", h_min_vals[i]);
    }
    printf("\n");
    std::vector<float> h_max_vals(N); 
    CUDA_CHECK(cudaMemcpy(h_max_vals.data(), d_max_vals,
                            N * sizeof(float),
                            cudaMemcpyDeviceToHost));
    printf("Max val: ");
    for (int i = 0; i < N; ++i) {
        printf("%f, ", h_max_vals[i]);
    }
    printf("\n");
  }
  std::vector<float> h_bin_widths(N); 
  CUDA_CHECK(cudaMemcpy(h_bin_widths.data(), d_bin_widths,
                          N * sizeof(float),
                          cudaMemcpyDeviceToHost));
  if (verbose) {
    printf("Bin widths: ");
    for (int i = 0; i < N; ++i) {
        printf("%f, ", h_bin_widths[i]);
    }
    printf("\n");
  }

  using ObjectiveT = decltype(objective);
  //using BinT = typename decltype(objective)::BinT;
  using BinT = ML::DT::CountBin;
  
  static constexpr int TPB = 256;

  std::vector<IdxT> h_row_ids(M);
  std::iota(h_row_ids.begin(), h_row_ids.end(), 0);

  std::vector<IdxT>  h_n_bins(N, max_n_bins);//N=num_proj
  // Quantiles (uniform splits in [0,1])
  // std::vector<DataT> h_quantiles(max_n_bins * N);
  // for (IdxT col = 0; col < N; ++col) {
  //     for (IdxT b = 0; b < max_n_bins; ++b) {
  //         h_quantiles[col * max_n_bins + b] = (b + 1) / float(max_n_bins);
  //     }
  // }
  std::vector<DataT> h_quantiles(max_n_bins * N);
  for (IdxT col = 0; col < N; ++col) {
      for (IdxT b = 0; b < max_n_bins; ++b) {
          h_quantiles[col * max_n_bins + b] = h_min_vals[col] + (b + 1) * h_bin_widths[col];
      }
  }
  if (verbose) {
    printf("Quantiles last value: ");
    for (int col = 0; col < N; ++col) {
        printf("%f, ", h_quantiles[(col + 1) * max_n_bins - 1]);
    }
    printf("\n");
  }

  // Define for sampling all features/projections
  std::vector<IdxT> h_colids (num_nodes * n_sampled_cols);
  for (int node = 0; node < num_nodes; ++node){
    for (int col = 0; col < n_sampled_cols; ++col){
      h_colids[node * n_sampled_cols + col] = col;
    }
  }

  std::vector<ML::DT::NodeWorkItem>    h_work_items(num_nodes); //divide the rows/number of samples

  // Split the dataset based on the number of nodes
  int base_chunk = M / num_nodes;
  int remainder  = M % num_nodes;  // leftover to distribute
  int start = 0;
  for (int node = 0; node < num_nodes; ++node) {
      int count = base_chunk + (node < remainder ? 1 : 0); // first 'remainder' chunks get +1
      h_work_items[node].instances.begin = start;
      h_work_items[node].instances.count = count;
      start += count;
      // std::cout << "Node [" << node << "] rows: " << count << std::endl;  
  }
  
  std::vector<int> blocks_per_node_row(num_nodes);
  int total_blocks_row = 0;
  for (int node = 0; node < num_nodes; ++node) {
      blocks_per_node_row[node] = static_cast<int>((h_work_items[node].instances.count + TPB - 1) / TPB);
      total_blocks_row += blocks_per_node_row[node];
  }
  // std::cout << "Total blocks row: " << total_blocks_row << std::endl;  

  std::vector<ML::DT::WorkloadInfo<IdxT>> h_workload_info(total_blocks_row);
  int offset = 0;
  for (int nid = 0; nid < num_nodes; ++nid) {
      int nb = blocks_per_node_row[nid];
      for (int b = 0; b < nb; ++b) {
          auto& info   = h_workload_info[offset + b];
          info.nodeid         = nid;
          info.large_nodeid   = nid;    // kernel needs this for histogram slab indexing
          info.offset_blockid = b;
          info.num_blocks     = nb;
      }
      offset += nb;
  }

  LabelT*   d_labels          = nullptr;
  IdxT*     d_row_ids         = nullptr;
  DataT*    d_quantiles       = nullptr;
  IdxT*     d_n_bins_array    = nullptr;
  IdxT*     d_colids          = nullptr;
  ML::DT::NodeWorkItem*       d_work_items      = nullptr;
  ML::DT::WorkloadInfo<IdxT>* d_workload_info   = nullptr;
  int*      d_done_count      = nullptr;
  int*      d_mutex           = nullptr;
  BinT*     d_histograms      = nullptr;
  ML::DT::Split<DataT,IdxT>* d_splits = nullptr;


  CUDA_CHECK(cudaMalloc(&d_labels,        M * sizeof(LabelT)));
  CUDA_CHECK(cudaMalloc(&d_row_ids,       M * sizeof(IdxT)));
  CUDA_CHECK(cudaMalloc(&d_quantiles,     max_n_bins * N * sizeof(DataT)));
  CUDA_CHECK(cudaMalloc(&d_n_bins_array,  N * sizeof(IdxT)));
  CUDA_CHECK(cudaMalloc(&d_colids,        num_nodes * n_sampled_cols * sizeof(IdxT)));
  CUDA_CHECK(cudaMalloc(&d_work_items,    num_nodes * sizeof(ML::DT::NodeWorkItem)));
  CUDA_CHECK(cudaMalloc(&d_workload_info, total_blocks_row * sizeof(ML::DT::WorkloadInfo<IdxT>)));
  CUDA_CHECK(cudaMalloc(&d_done_count,    num_nodes * n_sampled_cols * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_mutex,         num_nodes * sizeof(int)));
  
  CUDA_CHECK(cudaMemset(d_labels, 0, M * sizeof(LabelT)));
  CUDA_CHECK(cudaMemset(d_row_ids, 0, M * sizeof(IdxT)));
  CUDA_CHECK(cudaMemset(d_quantiles, 0, max_n_bins * N * sizeof(DataT)));
  CUDA_CHECK(cudaMemset(d_n_bins_array, 0, N * sizeof(IdxT)));
  CUDA_CHECK(cudaMemset(d_colids, 0, num_nodes * n_sampled_cols * sizeof(IdxT)));
  CUDA_CHECK(cudaMemset(d_work_items, 0, num_nodes * sizeof(ML::DT::NodeWorkItem)));
  CUDA_CHECK(cudaMemset(d_workload_info, 0, total_blocks_row * sizeof(ML::DT::WorkloadInfo<IdxT>)));
  CUDA_CHECK(cudaMemset(d_done_count, 0, num_nodes * n_sampled_cols * sizeof(int)));
  CUDA_CHECK(cudaMemset(d_mutex, 0, num_nodes * sizeof(int)));

  // Histograms shape: [nodes][features][bins][classes]
    size_t hist_elems = size_t(num_nodes) * n_sampled_cols 
                       * max_n_bins * num_classes;
    CUDA_CHECK(cudaMalloc(&d_histograms, hist_elems * sizeof(BinT)));
    CUDA_CHECK(cudaMemset(d_histograms, 0, hist_elems * sizeof(BinT)));

    CUDA_CHECK(cudaMalloc(&d_splits,     num_nodes * sizeof(ML::DT::Split<DataT,IdxT>)));
    CUDA_CHECK(cudaMemset(d_splits, 0, num_nodes * sizeof(ML::DT::Split<DataT,IdxT>)));


  // 6. Copy host → device
    //
    CUDA_CHECK(cudaMemcpy(d_labels,       h_labels.data(),   M * sizeof(LabelT),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_row_ids,      h_row_ids.data(),  M * sizeof(IdxT),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_quantiles,    h_quantiles.data(),
                          max_n_bins * N * sizeof(DataT),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_n_bins_array, h_n_bins.data(),
                          N * sizeof(IdxT),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_colids,       h_colids.data(),
                          num_nodes * n_sampled_cols * sizeof(IdxT),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_work_items,   h_work_items.data(),
                          num_nodes * sizeof(ML::DT::NodeWorkItem),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_workload_info,h_workload_info.data(),
                          total_blocks_row * sizeof(ML::DT::WorkloadInfo<IdxT>),
                          cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemset(d_done_count, 0, num_nodes * n_sampled_cols * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_mutex,      0, num_nodes * sizeof(int)));

  
  // 7. Set up device‐side structs
    //
    ML::DT::Dataset<DataT,LabelT,IdxT>   d_dataset;
    ML::DT::Quantiles<DataT,IdxT>        d_quants;

    //d_dataset.data           = d_data;
    d_dataset.data           = d_projected;
    d_dataset.labels         = d_labels;
    d_dataset.row_ids        = d_row_ids;
    d_dataset.M              = M;
    d_dataset.N              = N;
    d_dataset.n_sampled_cols = n_sampled_cols;

    d_quants.quantiles_array = d_quantiles;
    d_quants.n_bins_array    = d_n_bins_array;

    dim3 grid(total_blocks_row, n_sampled_cols, 1);

    //to calculate smem_bytes
    size_t shared_hist = max_n_bins * num_classes * sizeof(BinT);
    size_t shared_q    = max_n_bins * sizeof(DataT);
    size_t shared_done = max_n_bins * sizeof(int);
    size_t smem_bytes = shared_hist + shared_q + shared_done + 128;

  // --- host‐side setup ---
  // 1) Allocate a big array for every (node × feature) Split:
  int total_feat_splits = num_nodes * n_sampled_cols;
  ML::DT::Split<DataT,IdxT>* d_all_splits = nullptr;
  CUDA_CHECK(cudaMalloc(&d_all_splits,
          total_feat_splits * sizeof(ML::DT::Split<DataT,IdxT>)));
  CUDA_CHECK(cudaMemset(d_all_splits, 0, total_feat_splits * sizeof(ML::DT::Split<DataT,IdxT>)));

    // Launch kernel
    //
    cudaPointerAttributes attr;
    cudaPointerGetAttributes(&attr, d_projected);
    assert(attr.type == cudaMemoryTypeDevice);

    warmup1<<<1, 1>>>();
    cudaDeviceSynchronize();

    auto startA = std::chrono::high_resolution_clock::now();
    ML::DT::computeSplitKernel<DataT, LabelT, IdxT, TPB, ObjectiveT, BinT><<<grid, TPB, smem_bytes>>>(
        d_histograms,
        max_n_bins,
        max_depth,
        min_samples_split,
        max_leaves,
        d_dataset,
        d_quants,
        d_work_items,
        colStart,
        d_colids,
        d_done_count,
        d_mutex,
        d_splits,
        d_all_splits,
        objective,
        treeid,
        d_workload_info,
        seed
    );
    CUDA_CHECK(cudaDeviceSynchronize());
    auto endA = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> durationA = endA - startA;
    elapsed_ms = durationA.count() + duration0.count() + duration1.count();
  //
    // 10. Fetch and print results
    //
    // std::vector<ML::DT::Split<DataT,IdxT>> h_splits(num_nodes);
    CUDA_CHECK(cudaMemcpy(h_splits.data(), d_splits,
                          num_nodes * sizeof(ML::DT::Split<DataT,IdxT>),
                          cudaMemcpyDeviceToHost));

    // for (IdxT nid = 0; nid < num_nodes; ++nid) {
    //     auto &sp = h_splits[nid];
    //       //std::cout << "Best split for node " << nid << ":\n";
    //       std::cout << "  Projection: " << sp.colid << "\n";
    //       std::cout << "  Gain: " << sp.best_metric_val << "\n";
    //       std::cout << "  Threshold: " << sp.quesval << "\n";
          
    // }

    // 3) Copy back and print every split:
    std::vector<ML::DT::Split<DataT,IdxT>> h_all_splits(total_feat_splits);
    CUDA_CHECK(cudaMemcpy(
        h_all_splits.data(),
        d_all_splits,
        total_feat_splits * sizeof(ML::DT::Split<DataT,IdxT>),
        cudaMemcpyDeviceToHost));
    if (verbose) { 
      for (int nid = 0; nid < num_nodes; ++nid) {
        for (int f = 0; f < n_sampled_cols; ++f) {
          int idx = nid * n_sampled_cols + f;
          auto &S = h_all_splits[idx];
          printf("Node %d, Projection %d => gain = %f, thr = %f\n",
                nid, S.colid, S.best_metric_val, S.quesval);
        }
      }
    }

 
    // 11. Cleanup

    cudaFree(d_labels);
    cudaFree(d_row_ids);
    cudaFree(d_quantiles);
    cudaFree(d_n_bins_array);
    cudaFree(d_colids);
    cudaFree(d_work_items);
    cudaFree(d_workload_info);
    cudaFree(d_done_count);
    cudaFree(d_mutex);
    cudaFree(d_histograms);
    cudaFree(d_splits);
    cudaFree(d_all_splits);
    cudaFree(d_projected);//Free here since gini splits need to use the projected values, not right after apply projection

}

//For float, int, int
template void GiniGainsBestSplits<float, int, int>(
    float*,                                                         // d_projected
    const std::vector<int, std::allocator<int>>&,                   // h_labels
    int, int, int, int, int, int, int, int,                        // M … max_leaves
    unsigned long,                                                  // seed  ⚠  <- must match call
    int, int, int, int,                                             // treeid … min_leaf
    ML::DT::GiniObjectiveFunction<float, int, int>&,                // objective
    std::vector<ML::DT::Split<float, int>,
    std::allocator<ML::DT::Split<float, int>>>&,        // h_splits  (by ref, with allocator)
    double&,                                                        // elapsed_ms
    bool); 
//For double, int, int 
// #define INSTANTIATE_GINI_SPLITS(DataT, LabelT, IdxT) \
// template void GiniGainsBestSplits<DataT, LabelT, IdxT>( \
//   const std::vector<DataT>&, \
//   const std::vector<LabelT>&, \
//   IdxT, IdxT, IdxT, IdxT, IdxT, IdxT, IdxT, IdxT, \
//   uint64_t, IdxT, IdxT, IdxT, IdxT, \
//   ML::DT::GiniObjectiveFunction<DataT, LabelT, IdxT>&);

// INSTANTIATE_GINI_SPLITS(float, int, int)
// INSTANTIATE_GINI_SPLITS(double, int, int)
// INSTANTIATE_GINI_SPLITS(float, int, int64_t)

