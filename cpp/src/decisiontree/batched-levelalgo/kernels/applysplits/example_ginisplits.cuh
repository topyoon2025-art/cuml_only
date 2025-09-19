
#pragma once
#include "../../objectives.cuh"


template <typename DataT, typename LabelT, typename IdxT>
//void GiniGainsBestSplits (const std::vector<DataT>& h_data,
void GiniGainsBestSplits (DataT* d_projected,
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
                          ML::DT::GiniObjectiveFunction<DataT, LabelT, IdxT>& objective,
                          std::vector<ML::DT::Split<DataT, IdxT>>& h_splits,
                          double& elapsed_ms,
                          bool verbose);

