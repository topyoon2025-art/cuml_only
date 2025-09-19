#include <cuda_runtime.h>
#include <iostream>
#include "../builder_kernels_impl.cuh" // includes computeSplitKernel and related structs
#include "../../dataset.h"  // Make sure this path is correct and that dataset.h defines Dataset
#include "../../objectives.cuh"
#include <unordered_set>
#include "utils.hpp"

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
ML::DT::Split<float, int> GiniGainsBestSplits (const std::vector<DataT>& h_data,
                          const std::vector<LabelT>& h_labels,
                          const IdxT M,    // number of samples
                          const IdxT N,       // total number of features
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
                          ML::DT::GiniObjectiveFunction<DataT, LabelT, IdxT>& objective) {
  // Implementation of GiniGainsBestSplits
  // This function will compute the best splits for the Gini impurity
  // using the provided data and parameters.
  // You will need to implement the logic for finding the best splits
  // based on the Gini impurity criterion.
  
  using ObjectiveT = decltype(objective);
  //using BinT = typename decltype(objective)::BinT;
  using BinT = ML::DT::CountBin;
  
  static constexpr int TPB = 128;

  std::vector<IdxT>   h_row_ids(M);
  std::iota(h_row_ids.begin(), h_row_ids.end(), 0);

  // Quantiles (uniform splits in [0,1])
  std::vector<DataT> h_quantiles(max_n_bins * N);
  std::vector<IdxT>  h_n_bins(N, max_n_bins);
  for (IdxT col = 0; col < N; ++col) {
    std::vector<DataT> col_data(M);
    for (IdxT row = 0; row < M; ++row) {
        col_data[row] = h_data[row * N + col];  // assuming row-major layout
    }

    std::sort(col_data.begin(), col_data.end());

    for (IdxT b = 0; b < max_n_bins; ++b) {
        float quantile_pos = (b + 1) / float(max_n_bins + 1);  // avoid max edge
        IdxT idx = static_cast<IdxT>(quantile_pos * M);
        idx = std::min(idx, M - 1);  // clamp to valid index
        h_quantiles[col * max_n_bins + b] = col_data[idx];
    }
  }

  printf("Quantiles:\n");
  for (size_t i = 0; i < h_quantiles.size(); ++i) {
      printf("  h_quantiles[%zu] = %f\n", i, h_quantiles[i]);
  }

    // Define for sampling all features
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
      std::cout << "Node [" << node << "] blocks: " << count << std::endl;  
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

  DataT*    d_data            = nullptr;
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

  CUDA_CHECK(cudaMalloc(&d_data,          M * N * sizeof(DataT)));
  CUDA_CHECK(cudaMalloc(&d_labels,        M * sizeof(LabelT)));
  CUDA_CHECK(cudaMalloc(&d_row_ids,       M * sizeof(IdxT)));
  CUDA_CHECK(cudaMalloc(&d_quantiles,     max_n_bins * N * sizeof(DataT)));
  CUDA_CHECK(cudaMalloc(&d_n_bins_array,  N * sizeof(IdxT)));
  CUDA_CHECK(cudaMalloc(&d_colids,        num_nodes * n_sampled_cols * sizeof(IdxT)));
  CUDA_CHECK(cudaMalloc(&d_work_items,    num_nodes * sizeof(ML::DT::NodeWorkItem)));
  CUDA_CHECK(cudaMalloc(&d_workload_info, total_blocks_row * sizeof(ML::DT::WorkloadInfo<IdxT>)));
  CUDA_CHECK(cudaMalloc(&d_done_count,    num_nodes * n_sampled_cols * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_mutex,         num_nodes * sizeof(int)));

  // Histograms shape: [nodes][features][bins][classes]
    size_t hist_elems = size_t(num_nodes) * n_sampled_cols 
                       * max_n_bins * num_classes;
    CUDA_CHECK(cudaMalloc(&d_histograms, hist_elems * sizeof(BinT)));
    CUDA_CHECK(cudaMalloc(&d_splits,     num_nodes * sizeof(ML::DT::Split<DataT,IdxT>)));

  // 6. Copy host → device
    //
    CUDA_CHECK(cudaMemcpy(d_data,         h_data.data(),     M * N * sizeof(DataT),
                          cudaMemcpyHostToDevice));
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
    CUDA_CHECK(cudaMemcpy(d_workload_info,      h_workload_info.data(),
                          total_blocks_row * sizeof(ML::DT::WorkloadInfo<IdxT>),
                          cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemset(d_done_count, 0, num_nodes * n_sampled_cols * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_mutex,      0, num_nodes * sizeof(int)));

  
  // 7. Set up device‐side structs
    //
    ML::DT::Dataset<DataT,LabelT,IdxT>   d_dataset;
    ML::DT::Quantiles<DataT,IdxT>        d_quants;

    d_dataset.data           = d_data;
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

    // Launch kernel
    //
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
          std::cout << "  Column: " << sp.colid << "\n";
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

    for (int nid = 0; nid < num_nodes; ++nid) {
      for (int f = 0; f < n_sampled_cols; ++f) {
        int idx = nid * n_sampled_cols + f;
        auto &S = h_all_splits[idx];
        printf("Node %d, Column %d => gain = %f, thr = %f\n",
              nid, S.colid, S.best_metric_val, S.quesval);
      }
    }

    //
    // 11. Cleanup
    //
    cudaFree(d_data);
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
    
    return h_splits[0]; // Return the first split as an example

 

}

//For float, int, int
template ML::DT::Split<float, int> GiniGainsBestSplits<float, int, int>(
  const std::vector<float>&,
  const std::vector<int>&,
  int, int, int, int, int, int, int, int,
  uint64_t, int, int, int, int,
  ML::DT::GiniObjectiveFunction<float, int, int>&);

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
