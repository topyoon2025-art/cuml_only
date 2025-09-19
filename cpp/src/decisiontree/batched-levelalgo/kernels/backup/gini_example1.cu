#include <cuda_runtime.h>
#include <vector>
#include <numeric>
#include <iostream>

// Your project headers
#include "dataset.hpp"
#include "quantiles.hpp"
#include "work_items.hpp"
#include "workload_info.hpp"
#include "split.hpp"
#include "objective.hpp"
#include "kernel.hpp"
#include "gini_example.cuh"

using DataT      = float;
using LabelT     = int;
using IdxT       = int;
using BinT       = Bin<IdxT>;                          
using ObjectiveT = Gini<ObjectiveT, DataT, IdxT>;     

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
    //
    // 1. Problem dimensions
    //
    const IdxT N               = 1024;    // number of samples
    const IdxT M               = 4;       // total number of features
    const IdxT n_sampled_cols  = 2;       // two features per node
    const IdxT num_nodes       = 2;       // two nodes in this example
    const IdxT max_n_bins      = 8;
    const IdxT max_depth       = 3;
    const IdxT min_samples_split = 2;
    const IdxT max_leaves      = 4;
    const uint64_t seed        = 2025ULL;
    const IdxT treeid          = 0;
    const IdxT colStart        = 0;

    //
    // 2. Host data arrays
    //
    std::vector<DataT>  h_data(N * M);
    std::vector<LabelT> h_labels(N);
    std::vector<IdxT>   h_row_ids(N);
    std::iota(h_row_ids.begin(), h_row_ids.end(), 0);

    for (IdxT i = 0; i < N; ++i) {
        h_labels[i] = static_cast<LabelT>(i % ObjectiveT::NumClasses());
        for (IdxT j = 0; j < M; ++j) {
            h_data[i * M + j] = (i * 1.0f + j) / (N + M);
        }
    }

    //
    // 3. Quantiles (uniform splits in [0,1])
    //
    std::vector<DataT> h_quantiles(max_n_bins * M);
    std::vector<IdxT>  h_n_bins(M, max_n_bins);
    for (IdxT col = 0; col < M; ++col) {
        for (IdxT b = 0; b < max_n_bins; ++b) {
            h_quantiles[col * max_n_bins + b] = (b + 1) / float(max_n_bins);
        }
    }

    //
    // 4. Define two nodes, each sampling two features
    //    node 0 will test features {1,3}, node 1 will test {0,2}
    //
    std::vector<IdxT> h_colids = {
      /* node0 */ 1, 3,
      /* node1 */ 0, 2
    };

    std::vector<NodeWorkItem>    h_work_items(num_nodes);
    std::vector<WorkloadInfo<IdxT>> h_wl_info(num_nodes);

    // Split the dataset in half between the two nodes
    h_work_items[0] = {/* begin */ 0,     /* count */ N/2};
    h_work_items[1] = {/* begin */ N/2,  /* count */ N - N/2};

    for (IdxT nid = 0; nid < num_nodes; ++nid) {
        h_wl_info[nid].nodeid         = nid;
        h_wl_info[nid].large_nodeid   = nid;
        h_wl_info[nid].offset_blockid = 0;
        h_wl_info[nid].num_blocks     = 1;
    }

    //
    // 5. Allocate device memory
    //
    DataT*    d_data            = nullptr;
    LabelT*   d_labels          = nullptr;
    IdxT*     d_row_ids         = nullptr;
    DataT*    d_quantiles       = nullptr;
    IdxT*     d_n_bins_array    = nullptr;
    IdxT*     d_colids          = nullptr;
    NodeWorkItem*      d_work_items = nullptr;
    WorkloadInfo<IdxT>* d_wl_info   = nullptr;
    int*      d_done_count      = nullptr;
    int*      d_mutex           = nullptr;
    BinT*     d_histograms      = nullptr;
    Split<DataT,IdxT>* d_splits = nullptr;

    CUDA_CHECK(cudaMalloc(&d_data,          N * M * sizeof(DataT)));
    CUDA_CHECK(cudaMalloc(&d_labels,        N * sizeof(LabelT)));
    CUDA_CHECK(cudaMalloc(&d_row_ids,       N * sizeof(IdxT)));
    CUDA_CHECK(cudaMalloc(&d_quantiles,     max_n_bins * M * sizeof(DataT)));
    CUDA_CHECK(cudaMalloc(&d_n_bins_array,  M * sizeof(IdxT)));
    CUDA_CHECK(cudaMalloc(&d_colids,        num_nodes * n_sampled_cols * sizeof(IdxT)));
    CUDA_CHECK(cudaMalloc(&d_work_items,    num_nodes * sizeof(NodeWorkItem)));
    CUDA_CHECK(cudaMalloc(&d_wl_info,       num_nodes * sizeof(WorkloadInfo<IdxT>)));
    CUDA_CHECK(cudaMalloc(&d_done_count,    num_nodes * n_sampled_cols * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_mutex,         num_nodes * sizeof(int)));

    // Histograms shape: [nodes][features][bins][classes]
    size_t hist_elems = size_t(num_nodes) * n_sampled_cols 
                       * max_n_bins * ObjectiveT::NumClasses();
    CUDA_CHECK(cudaMalloc(&d_histograms, hist_elems * sizeof(BinT)));
    CUDA_CHECK(cudaMalloc(&d_splits,     num_nodes * sizeof(Split<DataT,IdxT>)));

    //
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
                          num_nodes * sizeof(NodeWorkItem),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_wl_info,      h_wl_info.data(),
                          num_nodes * sizeof(WorkloadInfo<IdxT>),
                          cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemset(d_done_count, 0, num_nodes * n_sampled_cols * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_mutex,      0, num_nodes * sizeof(int)));

    //
    // 7. Set up device‐side structs
    //
    Dataset<DataT,LabelT,IdxT>   d_dataset;
    Quantiles<DataT,IdxT>        d_quants;

    d_dataset.data           = d_data;
    d_dataset.labels         = d_labels;
    d_dataset.row_ids        = d_row_ids;
    d_dataset.M              = M;
    d_dataset.N              = N;
    d_dataset.n_sampled_cols = n_sampled_cols;

    d_quants.quantiles_array = d_quantiles;
    d_quants.n_bins_array    = d_n_bins_array;

    //
    // 8. Launch parameters
    //
    constexpr int TPB = 128;
    dim3 grid(num_nodes, n_sampled_cols);
    dim3 block(TPB);

    size_t shared_hist = max_n_bins * ObjectiveT::NumClasses() * sizeof(BinT);
    size_t shared_q    = max_n_bins * sizeof(DataT);
    size_t shared_done = max_n_bins * sizeof(int);
    size_t smem_bytes = shared_hist + shared_q + shared_done + 128; 

    //
    // 9. Launch kernel
    //
    computeSplitKernel<<<grid, block, smem_bytes>>>(
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
        ObjectiveT(),
        treeid,
        d_wl_info,
        seed
    );
    CUDA_CHECK(cudaDeviceSynchronize());

    //
    // 10. Fetch and print results
    //
    std::vector<Split<DataT,IdxT>> h_splits(num_nodes);
    CUDA_CHECK(cudaMemcpy(h_splits.data(), d_splits,
                          num_nodes * sizeof(Split<DataT,IdxT>),
                          cudaMemcpyDeviceToHost));

    for (IdxT nid = 0; nid < num_nodes; ++nid) {
        auto &sp = h_splits[nid];
        std::cout << "Node " << nid
                  << " -> feature " << sp.feature
                  << ", bin " << sp.bin
                  << ", gain " << sp.gain
                  << "\n";
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
    cudaFree(d_wl_info);
    cudaFree(d_done_count);
    cudaFree(d_mutex);
    cudaFree(d_histograms);
    cudaFree(d_splits);

    return 0;
}