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

  constexpr int num_rows = 150; //M, number of samples
  int n_blocks_per_node = (num_rows + TPB - 1) / TPB;
  printf("blocks per node: %d\n", n_blocks_per_node);

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
      h_labels[i] = static_cast<LabelT>(i % objective.NumClasses());
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
  std::vector<ML::DT::WorkloadInfo<IdxT>> h_wl_info(num_nodes);

  // Split the dataset in half between the two nodes
  //for struct
  h_work_items[0] = {/* begin */ 0,     /* count */ N/2}; //for integer, round down
  h_work_items[1] = {/* begin */ N/2,  /* count */ N - N/2};






  ML::DT::Dataset<DataT, LabelT, IdxT> h_dataset;
  


  std::vector<DataT>    h_data(num_rows * num_features);
  std::vector<LabelT>   h_labels(num_rows);
  std::vector<IdxT>     h_row_ids(num_rows);

  // Populate feature matrix (column-major)
  for (IdxT col = 0; col < num_features; ++col) {
    for (IdxT row = 0; row < num_rows; ++row) {
      h_data[col * num_rows + row] = static_cast<DataT>(rand() % 100) / 100.0f;
    }
  }
  
  // Populate labels and row_ids
  for (IdxT i = 0; i < num_rows; ++i) {
    h_labels[i] = static_cast<LabelT>(i % num_classes);
    h_row_ids[i] = i;
  }
  printf("h_data size: %zu, h_labels size: %zu, h_row_ids size: %zu\n",
         h_data.size(), h_labels.size(), h_row_ids.size());
  printf("first five h_labels: ");
  for (int i = 0; i < 5 && i < h_labels.size(); ++i) {
    printf("%d ", h_labels[i]);
  }
  printf("\n");

    // Allocate and initialize dataset
  
  DataT*   d_data   = nullptr;
  LabelT*  d_labels = nullptr;
  IdxT*    d_row_ids = nullptr;

  cudaMallocManaged(&d_data, sizeof(DataT) * num_rows * num_features); //float
  cudaMallocManaged(&d_labels, sizeof(LabelT) * num_rows); //int
  cudaMallocManaged(&d_row_ids, sizeof(IdxT) * num_rows); //int
  cudaMemcpy(d_data, h_data.data(), sizeof(DataT) * num_rows * num_features, cudaMemcpyHostToDevice);
  cudaMemcpy(d_labels, h_labels.data(), sizeof(LabelT) * num_rows, cudaMemcpyHostToDevice);
  cudaMemcpy(d_row_ids, h_row_ids.data(), sizeof(IdxT) * num_rows, cudaMemcpyHostToDevice);


  // Set dataset metadata
  h_dataset.M = num_rows;
  h_dataset.N = num_features;
  h_dataset.n_sampled_cols = num_sampled_cols;
  h_dataset.num_outputs = num_classes;
  h_dataset.row_ids = d_row_ids;
  h_dataset.data = d_data;
  h_dataset.labels = d_labels;


  // Quantiles
  ML::DT::Quantiles<DataT, IdxT> quantiles;
  cudaMallocManaged(&quantiles.quantiles_array, sizeof(DataT) * num_features * n_bins);
  cudaMallocManaged(&quantiles.n_bins_array, sizeof(IdxT) * num_features);
  for (int f = 0; f < num_features; ++f) {
    quantiles.n_bins_array[f] = n_bins;
    for (int b = 0; b < n_bins; ++b) {
      quantiles.quantiles_array[f * n_bins + b] = static_cast<DataT>(b) / n_bins;
    }
  }

  //This is for single node  
  // Work items
  ML::DT::NodeWorkItem* work_items;
  cudaMallocManaged(&work_items, sizeof(ML::DT::NodeWorkItem) * n_blocks_per_node);

  // Workload info
  ML::DT::WorkloadInfo<IdxT>* workload_info;
  cudaMallocManaged(&workload_info, sizeof(ML::DT::WorkloadInfo<IdxT>) * n_blocks_per_node);

  int range_start = 0;
  int range_len = num_rows;
  int nid = 0;
  for (IdxT i = 0; i < n_blocks_per_node; ++i) {
    IdxT start = range_start + i * TPB;
    IdxT cnt   = (i+1 == n_blocks_per_node)
                  ? (range_len - i * TPB)
                  : TPB;
    work_items[i].instances.begin = start;
    work_items[i].instances.count = cnt;
    workload_info[i].nodeid         = nid;
    workload_info[i].large_nodeid   = n_blocks_per_node - 2;
    workload_info[i].offset_blockid = i;
    workload_info[i].num_blocks     = n_blocks_per_node;
  }
  printf("Workload info for node %d:\n", nid);
  for (int i = 0; i < n_blocks_per_node; ++i) {
    printf("  Block %d: begin=%d, count=%d\n",
           i, work_items[i].instances.begin, work_items[i].instances.count);
  }

  // Output splits
  ML::DT::Split<DataT, IdxT>* splits;
  cudaMallocManaged(&splits, sizeof(ML::DT::Split<DataT, IdxT>) * num_nodes);

  // Histogram buffer
  ML::DT::GiniObjectiveFunction<DataT,LabelT,IdxT> objective(num_classes, min_samples_leaf);

  ML::DT::GiniObjectiveFunction<DataT,LabelT,IdxT>::BinT* d_histograms;
  cudaMallocManaged(&d_histograms, sizeof(ML::DT::GiniObjectiveFunction<DataT,LabelT,IdxT>::BinT) * num_nodes * num_features * n_bins * num_classes);

  // Other buffers
  int* done_count;
  int* mutex;
  cudaMallocManaged(&done_count, sizeof(int) * num_nodes * num_sampled_cols);
  cudaMallocManaged(&mutex, sizeof(int) * num_nodes);

  cudaMemset(done_count, 0, sizeof(int) * num_nodes * num_sampled_cols);
  cudaMemset(mutex,      0, sizeof(int) * num_nodes);

  cudaDeviceSynchronize();


  // Launch kernel
  size_t smem_size = sizeof(ML::DT::GiniObjectiveFunction<DataT,LabelT,IdxT>::BinT) * n_bins * num_classes + sizeof(DataT) * n_bins + sizeof(int);
  dim3 grid(n_blocks_per_node, num_sampled_cols);
  printf("Launching grid (%d, %d)\n", grid.x, grid.y);
  ML::DT::computeSplitKernel<DataT, LabelT, IdxT, TPB, ObjectiveT, BinT><<<grid, TPB, smem_size>>>(
    d_histograms, n_bins, 10, 2, 10, h_dataset, quantiles, work_items, 0,
    /* colids= */ nullptr, done_count, mutex, splits, objective, /* colStart= */ 0, workload_info, 12345);
  //pass nullptr for now to process all features.

  cudaDeviceSynchronize();

  // Print result
  std::cout << "Best split for node 0:\n";
  std::cout << "  Column: " << splits[0].colid << "\n";
  std::cout << "  Threshold: " << splits[0].quesval << "\n";
  std::cout << "  Gain: " << splits[0].best_metric_val << "\n";

  std::cout << "Best split for node 1:\n";
  std::cout << "  Column: " << splits[1].colid << "\n";
  std::cout << "  Threshold: " << splits[1].quesval << "\n";
  std::cout << "  Gain: " << splits[1].best_metric_val << "\n";

  // Cleanup
  cudaFree(d_data);
  cudaFree(d_labels);
  cudaFree(d_row_ids);
  cudaFree(quantiles.quantiles_array);
  cudaFree(quantiles.n_bins_array);
  cudaFree(work_items);
  cudaFree(workload_info);
  cudaFree(splits);
  cudaFree(d_histograms);
  cudaFree(done_count);
  cudaFree(mutex);

  return 0;
}
