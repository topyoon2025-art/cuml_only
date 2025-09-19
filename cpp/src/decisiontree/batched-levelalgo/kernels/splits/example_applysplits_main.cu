#include <stdio.h>
#include <cstdint>
#include <iostream>
#include <vector>
#include <cstdlib>   
#include <ctime>    
#include <random>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <iomanip>
#include <chrono>
#include <fstream>
#include <string>

#include "../../objectives.cuh"
#include "../../builder.cuh"
#include "example_ginisplits.cuh"
#include "utils.hpp"
#include "cuml/tree/decisiontree.hpp"

#include <raft/core/handle.hpp>


raft::handle_t handle;
// cudaStream_t stream = raft::get_stream(handle);


using DataT = float;
using LabelT = int;
using IdxT = int;
using ObjectiveT = ML::DT::GiniObjectiveFunction<DataT, LabelT, IdxT>;

std::vector<DataT> h_data = {
    0.6119, 0.4561, 0.7852, 0.5924, 0.1395, 0.2912, 0.3664, 0.5142, 0.6075, 0.1705,
    0.0651, 0.9489, 0.4560, 0.6842, 0.1220, 0.4952, 0.0344, 0.9093, 0.2588, 0.8324
};

std::vector<LabelT> h_labels = {
    0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
    0, 1, 0, 1, 0, 1, 0, 1, 0, 1
};

//int main (int argc, char* argv[]) {
int main () {


    //DecisionTreeParams
    ML::DT::DecisionTreeParams params;
    params.max_depth             = 1;       // Only allow root to split once
    params.max_leaves            = 2;       // One split → two leaves
    params.max_features          = 1.0f;    // Use all features/projections
    params.max_n_bins            = 4;      // Keep binning simple
    params.min_samples_leaf      = 1;       // Allow smallest leaf
    params.min_samples_split     = 2;       // Allow split with just 2 samples
    params.split_criterion       = ML::CRITERION::GINI;  // Or ENTROPY/MSE depending on task, GINI = 0 from Enum
    params.min_impurity_decrease = 0.0f;    // Allow any split with positive gain
    params.max_batch_size        = 1;       // One node at a time for clarity, limits how many work_items are popped from the queue at once

    size_t max_nodes    = 3;  // 1 root + 2 children (left and right)
    size_t sampled_rows = 20;
  
    size_t num_rows = 20;
    size_t num_cols = 1;
    int num_outputs = 2;

    float* d_features;
    int* d_labels;
    cudaMalloc(&d_features, num_rows * num_cols * sizeof(float));
    cudaMalloc(&d_labels, num_rows * sizeof(int));
    cudaMemcpy(d_features, h_data.data(), sizeof(float) * num_rows, cudaMemcpyHostToDevice);
    cudaMemcpy(d_labels, h_labels.data(), sizeof(int) * num_rows, cudaMemcpyHostToDevice);

    std::vector<int> h_row_ids(num_rows);
    std::iota(h_row_ids.begin(), h_row_ids.end(), 0);  // Fill with 0..19
    int* d_row_ids;
    cudaMalloc(&d_row_ids, sizeof(int) * num_rows);
    cudaMemcpy(d_row_ids, h_row_ids.data(), sizeof(int) * num_rows, cudaMemcpyHostToDevice);

    std::vector<DataT> h_quantiles(params.max_n_bins * num_cols, 0.0f);
    std::vector<IdxT>  h_n_bins_array(num_cols, params.max_n_bins);
    for (IdxT col = 0; col < num_cols; ++col) {
        for (IdxT b = 0; b < params.max_n_bins; ++b) {
            h_quantiles[col * params.max_n_bins + b] = (b + 1) / float(params.max_n_bins);
        }
    }
    DataT* d_quantiles;
    IdxT* d_n_bins_array;
    cudaMalloc(&d_quantiles, sizeof(DataT) * h_quantiles.size());
    cudaMalloc(&d_n_bins_array, sizeof(IdxT) * h_n_bins_array.size());
    cudaMemcpy(d_quantiles, h_quantiles.data(), sizeof(DataT) * h_quantiles.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_n_bins_array, h_n_bins_array.data(), sizeof(IdxT) * h_n_bins_array.size(), cudaMemcpyHostToDevice);

    ML::DT::Dataset<DataT,LabelT,IdxT>   d_dataset;
    d_dataset.data           = d_features;
    d_dataset.labels         = d_labels;
    d_dataset.row_ids        = d_row_ids;
    d_dataset.M              = num_rows;
    d_dataset.N              = num_cols;
    d_dataset.n_sampled_rows  = num_rows;
    d_dataset.n_sampled_cols  = num_cols;

    d_dataset.n_sampled_cols = num_cols;

    ML::DT::Quantiles<DataT,IdxT>        d_quants;
    d_quants.quantiles_array = d_quantiles;
    d_quants.n_bins_array    = d_n_bins_array;


    ML::DT::NodeQueue<float, int> queue(params, max_nodes, sampled_rows/*or features.size()*/, num_outputs);
    //First one at depth 0 print
    auto tree0 = queue.GetTree();
    std::cout << "Tree depth: " << tree0->depth_counter << "\n";
    std::cout << "Leaf count: " << tree0->leaf_counter << "\n";
    std::cout << "Total nodes: " << tree0->sparsetree.size() << "\n";

    // // Builder instantiation (low-level)
    // ML::DT::Builder<ObjectiveT> builder(
    //     handle,
    //     stream,
    //     0,                // treeid
    //     42,               // seed
    //     params,
    //     d_features,
    //     d_labels,
    //     num_rows,
    //     num_cols,
    //     &d_row_ids,
    //     num_outputs,
    //     d_quants
    // );

    

    int num_nodes = 1;
    const int treeid          = 0;
    const int colStart        = 0;
    const int num_classes     = 2;
    const uint64_t seed       = 2025ULL;
    ML::DT::GiniObjectiveFunction<float, int, int> objective(num_classes, params.min_samples_leaf);

    auto splits = GiniGainsBestSplits<float, int, int>(h_data, //replace with col add flattened data
                                    h_labels,
                                    num_rows,    // number of samples
                                    num_cols,       // total number of features/projections in this case for oblique
                                    num_cols,       // Sample every feature
                                    num_nodes,       // two nodes in this example
                                    params.max_n_bins,
                                    params.max_depth,//not used
                                    params.min_samples_split,
                                    params.max_leaves,//not used
                                    seed,
                                    treeid,
                                    colStart,
                                    num_classes, // number of classes for classification, binary for this example
                                    params.min_samples_leaf,
                                    objective);
    std::vector<ML::DT::Split<float, int>> best_splits = { splits };
    printf("Best splits what I will use:\n");
    printf("  Split: col=%d, threshold=%f, gain=%f\n",
           best_splits[0].colid, best_splits[0].quesval, best_splits[0].best_metric_val);

    // // Compute splits
    // auto [h_splits, batch_size] = builder.doSplit(batch);

    auto tree = queue.GetTree();
    std::cout << "Tree depth: " << tree->depth_counter << "\n";
    std::cout << "Leaf count: " << tree->leaf_counter << "\n";
    std::cout << "Total nodes: " << tree->sparsetree.size() << "\n";
    // Pop root node
    auto batch = queue.Pop();
    // Apply splits to tree
    queue.Push(batch, best_splits.data());

    // Inspect tree state

    std::cout << "Tree depth: " << tree->depth_counter << "\n";
    std::cout << "Leaf count: " << tree->leaf_counter << "\n";
    std::cout << "Total nodes: " << tree->sparsetree.size() << "\n";



    const auto& ranges = queue.GetInstanceRanges();

    // Print them
    for (size_t i = 0; i < ranges.size(); ++i) {
    std::cout << "InstanceRange " << i << " {\n"
              << "  begin: " << ranges[i].begin << "\n"
              << "  depth: " << ranges[i].count << "\n"
              << "}\n";
}



 

    cudaFree(d_features);
    cudaFree(d_labels);
    cudaFree(d_row_ids);
    cudaFree(d_quantiles);
    cudaFree(d_n_bins_array);


//     //NodeWorkItem
//     // - A node index in the tree (idx)
//     // - Its depth in the tree (depth)
//     // - The range of data samples assigned to it (InstanceRange)

//     //  class NodeQueue {
//     //   using NodeT = SparseTreeNode<DataT, LabelT>;
//     //   const DecisionTreeParams params;
//     //   std::shared_ptr<DT::TreeMetaDataNode<DataT, LabelT>> tree;
//     //   std::vector<InstanceRange> node_instances_;
//     //   std::deque<NodeWorkItem> work_items_;

//     //  public:
//     //   NodeQueue(DecisionTreeParams params, size_t max_nodes, size_t sampled_rows, int num_outputs)
//     //     : params(params), tree(std::make_shared<DT::TreeMetaDataNode<DataT, LabelT>>())
//     //   {
//     //     tree->num_outputs = num_outputs;
//     //     tree->sparsetree.reserve(max_nodes);
//     //     tree->sparsetree.emplace_back(NodeT::CreateLeafNode(sampled_rows));
//     //     tree->leaf_counter  = 1;
//     //     tree->depth_counter = 0;
//     //     node_instances_.reserve(max_nodes);
//     //     node_instances_.emplace_back(InstanceRange{0, sampled_rows});
//     //     if (this->IsExpandable(tree->sparsetree.back(), 0)) {
//     //       work_items_.emplace_back(NodeWorkItem{0, 0, node_instances_.back()}); Node Index in the tree, Depth, Instance Range
//     //     }
//     //   }

// //    work_items_ = [
// //   {idx: 3, depth: 1, Instance: 10},
// //   {idx: 4, depth: 1, Instance: 20},
// //   {idx: 5, depth: 1, Instance: 30}
// // ]




//     /////////////////////////////////////////////////GiniGainsBestSplits//////////////////////////////////////////


//     // printf("num_rows: %d\n", num_rows);
//     // printf("num_proj: %d\n", num_proj);

//     // const int M               = num_rows;    // num_rows
//     // const int N               = num_proj;       // num_proj
//     // const int n_sampled_cols  = num_proj;       // Sample every feature if < n_sampled_cols = num_proj;
//     // const int num_nodes       = 1;       // 1 node in this example
//     // const int max_depth       = 3; //not used
//     // const int min_samples_split = 2; //not usedd_col_add_projected
//     // const int max_leaves      = 4; //not used
//     // const uint64_t seed       = 2025ULL;
//     // const int treeid          = 0;
//     // const int colStart        = 0;
//     // const int num_classes     = 2; // number of classes for classification, binary for this example
//     // const int min_samples_leaf  = 0; 
//     // ML::DT::GiniObjectiveFunction<float, int, int> objective(num_classes, min_samples_leaf);


//     // std::vector<int> h_labels(num_rows);
//     // for (int i = 0; i < num_rows; ++i) {
//     //     h_labels[i] = static_cast<int>(i % num_classes);
//     // }

    // GiniGainsBestSplits<float, int, int>(d_col_add_projected, //replace with col add flattened data
    //                                     h_labels,
    //                                     M,
    //                                     N,       // total number of features/projections in this case for oblique
    //                                     n_sampled_cols,       // Sample every feature
    //                                     num_nodes,       // two nodes in this example
    //                                     max_n_bins,
    //                                     max_depth,//not used
    //                                     min_samples_split,
    //                                     max_leaves,//not used
    //                                     seed,
    //                                     treeid,
    //                                     colStart,
    //                                     num_classes, // number of classes for classification, binary for this example
    //                                     min_samples_leaf,
    //                                     objective);

    // ///////////////////////////GiniGainsBestSplits//////////////////////////////////////////

    return 0;
    }
