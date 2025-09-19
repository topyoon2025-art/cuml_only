#include <cuda_runtime.h>
#include <iostream>
#include "builder_kernels_impl.cuh" // includes computeSplitKernel and related structs
#include "../dataset.h"  // Make sure this path is correct and that dataset.h defines Dataset
#include <unordered_set>


using DataT = float;
using LabelT = int;
using IdxT = int;
using ObjectiveT = ML::DT::GiniObjectiveFunction<DataT, LabelT, IdxT>;
using BinT = ML::DT::GiniObjectiveFunction<DataT,LabelT,IdxT>::BinT;

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

int main() {

  static constexpr int TPB = 128;

  const IdxT N               = 1024;    // number of samples
  const IdxT M               = 4;       // total number of features
  const IdxT n_sampled_cols  = 2;       // two features per node
  const IdxT num_nodes       = 2;       // two nodes in this example
  const IdxT max_n_bins      = 16;
  const IdxT max_depth       = 3;
  const IdxT min_samples_split = 2;
  const IdxT max_leaves      = 4;
  const uint64_t seed        = 2025ULL;
  const IdxT treeid          = 0;
  const IdxT colStart        = 0;
  const IdxT num_classes     = 2; // number of classes for classification
  const IdxT min_samples_leaf  = 2;

   ML::DT::GiniObjectiveFunction<DataT,LabelT,IdxT> objective(num_classes, min_samples_leaf);

  //
  // 2. Host data arrays
  //
  std::vector<DataT>  h_data(N * M);
  std::vector<LabelT> h_labels(N);
  std::vector<IdxT>   h_row_ids(N);
  std::iota(h_row_ids.begin(), h_row_ids.end(), 0);

  // 1) Fill labels (same as before)
  for (IdxT i = 0; i < N; ++i) {
      h_labels[i] = static_cast<LabelT>(i % num_classes);
  }

  // 2) Fill data in column-major order:
  //    contiguous stride = N (numRows)
  for (IdxT col = 0; col < M; ++col) {
      for (IdxT row = 0; row < N; ++row) {
          h_data[col * N + row] = (row * 1.0f + col) / (N + M);
      }
  }

  // 3. Quantiles (uniform splits in [0,1])
  //
  std::vector<DataT> h_quantiles(max_n_bins * M);
  std::vector<IdxT>  h_n_bins(M, max_n_bins);
  for (IdxT col = 0; col < M; ++col) {
      for (IdxT b = 0; b < max_n_bins; ++b) {
          h_quantiles[col * max_n_bins + b] = (b + 1) / float(max_n_bins);
      }
  }

    // 4. Define two nodes, each sampling two features
    //    node 0 will test features {1,3}, node 1 will test {0,2}
    //
    std::vector<IdxT> h_colids = {
      /* node0 */ 1, 3,
      /* node1 */ 0, 2
    };


  std::vector<ML::DT::NodeWorkItem>    h_work_items(num_nodes);
  

  // Split the dataset in half between the two nodes
  //for struct
  h_work_items[0].instances.begin = 0;
  h_work_items[0].instances.count = N/2;
  h_work_items[1].instances.begin = N/2;
  h_work_items[1].instances.count = N - N/2;
  
  std::cout << "Number of blocks per node: " << h_work_items[0].instances.count << ", " << h_work_items[1].instances.count << "\n";
  std::vector<int> n_blocks_per_node = {static_cast<int>((h_work_items[0].instances.count + TPB - 1) / TPB),
                                         static_cast<int>((h_work_items[1].instances.count + TPB - 1) / TPB)};
  const int total_blocks = n_blocks_per_node[0] + n_blocks_per_node[1];
  printf("blocks per node: %d\n", n_blocks_per_node[0]);
  printf("blocks per node: %d\n", n_blocks_per_node[1]);
  std::vector<ML::DT::WorkloadInfo<IdxT>> h_workload_info(total_blocks);
  int offset = 0;
  for (int nid = 0; nid < num_nodes; ++nid) {
      int nb = n_blocks_per_node[nid];
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

  CUDA_CHECK(cudaMalloc(&d_data,          N * M * sizeof(DataT)));
  CUDA_CHECK(cudaMalloc(&d_labels,        N * sizeof(LabelT)));
  CUDA_CHECK(cudaMalloc(&d_row_ids,       N * sizeof(IdxT)));
  CUDA_CHECK(cudaMalloc(&d_quantiles,     max_n_bins * M * sizeof(DataT)));
  CUDA_CHECK(cudaMalloc(&d_n_bins_array,  M * sizeof(IdxT)));
  CUDA_CHECK(cudaMalloc(&d_colids,        num_nodes * n_sampled_cols * sizeof(IdxT)));
  CUDA_CHECK(cudaMalloc(&d_work_items,    num_nodes * sizeof(ML::DT::NodeWorkItem)));
  CUDA_CHECK(cudaMalloc(&d_workload_info, (n_blocks_per_node[0] + n_blocks_per_node[1]) * sizeof(ML::DT::WorkloadInfo<IdxT>)));
  CUDA_CHECK(cudaMalloc(&d_done_count,    num_nodes * n_sampled_cols * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_mutex,         num_nodes * sizeof(int)));

  // Histograms shape: [nodes][features][bins][classes]
    size_t hist_elems = size_t(num_nodes) * n_sampled_cols 
                       * max_n_bins * num_classes;
    CUDA_CHECK(cudaMalloc(&d_histograms, hist_elems * sizeof(BinT)));
    CUDA_CHECK(cudaMalloc(&d_splits,     num_nodes * sizeof(ML::DT::Split<DataT,IdxT>)));

  // 6. Copy host → device
    //
    CUDA_CHECK(cudaMemcpy(d_data,         h_data.data(),     N * M * sizeof(DataT),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_labels,       h_labels.data(),   N * sizeof(LabelT),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_row_ids,      h_row_ids.data(),  N * sizeof(IdxT),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_quantiles,    h_quantiles.data(),
                          max_n_bins * M * sizeof(DataT),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_n_bins_array, h_n_bins.data(),
                          M * sizeof(IdxT),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_colids,       h_colids.data(),
                          num_nodes * n_sampled_cols * sizeof(IdxT),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_work_items,   h_work_items.data(),
                          num_nodes * sizeof(ML::DT::NodeWorkItem),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_workload_info,      h_workload_info.data(),
                          total_blocks * sizeof(ML::DT::WorkloadInfo<IdxT>),
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

    dim3 grid(total_blocks, n_sampled_cols, 1);

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

    // 9. Launch kernel
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

    return 0;

}
