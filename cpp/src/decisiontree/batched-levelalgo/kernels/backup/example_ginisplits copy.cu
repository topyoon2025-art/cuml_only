#include <cuda_runtime.h>
#include <iostream>
#include "../builder_kernels_impl.cuh" // includes computeSplitKernel and related structs
#include "../../dataset.h"  // Make sure this path is correct and that dataset.h defines Dataset
#include "../../objectives.cuh"
#include <unordered_set>
#include "utils.hpp"
#include <cub/cub.cuh>

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



template <typename DataT, typename LabelT, typename IdxT>
void GiniGainsBestSplits (DataT* d_projected,
                          const std::vector<LabelT>& h_labels,
                          const IdxT M,    // number of samples
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
                          bool verbose) {


  
  using ObjectiveT = decltype(objective);
  //using BinT = typename decltype(objective)::BinT;
  using BinT = ML::DT::CountBin;
  
  static constexpr int TPB = 256;

  std::vector<IdxT>   h_row_ids(M);
  std::iota(h_row_ids.begin(), h_row_ids.end(), 0);


  std::vector<IdxT>  h_n_bins(N, max_n_bins);//N=num_proj

  // Quantiles (uniform splits in [0,1])
  std::vector<DataT> h_quantiles(max_n_bins * N);
  for (IdxT col = 0; col < N; ++col) {
      for (IdxT b = 0; b < max_n_bins; ++b) {
          h_quantiles[col * max_n_bins + b] = (b + 1) / float(max_n_bins);
      }
  }

  printf("Quantiles: ");
  for (int i = 0; i < 5; ++i) {
      printf("%f, ", h_quantiles[i]);
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
      std::cout << "Node [" << node << "] rows: " << count << std::endl;  
  }
  
  std::vector<int> blocks_per_node_row(num_nodes);
  int total_blocks_row = 0;
  for (int node = 0; node < num_nodes; ++node) {
      blocks_per_node_row[node] = static_cast<int>((h_work_items[node].instances.count + TPB - 1) / TPB);
      total_blocks_row += blocks_per_node_row[node];
  }
  std::cout << "Total blocks row: " << total_blocks_row << std::endl;  

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
    std::cout << "Time taken for cuML Histogram and Gini gains: " << std::chrono::duration<double, std::milli>(endA - startA).count() << " ms";
    std::cout << std::endl;



  //
    // 10. Fetch and print results
    //
    std::vector<ML::DT::Split<DataT,IdxT>> h_splits(num_nodes);
    CUDA_CHECK(cudaMemcpy(h_splits.data(), d_splits,
                          num_nodes * sizeof(ML::DT::Split<DataT,IdxT>),
                          cudaMemcpyDeviceToHost));

    for (IdxT nid = 0; nid < num_nodes; ++nid) {
        auto &sp = h_splits[nid];
          std::cout << "Best split for node " << nid << ":\n";
          std::cout << "  Projection: " << sp.colid << "\n";
          std::cout << "  Threshold: " << sp.quesval << "\n";
          std::cout << "  Gain: " << sp.best_metric_val << "\n";
    }

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
  float*,
  const std::vector<int>&,
  int, int, int, int, int, int, int, int,
  uint64_t, int, int, int, int,
  ML::DT::GiniObjectiveFunction<float, int, int>&, bool);

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

