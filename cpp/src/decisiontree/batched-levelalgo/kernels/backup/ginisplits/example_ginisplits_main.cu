#include <string>
#include <cstdint>
#include "utils.hpp"

#include "../../objectives.cuh"
#include "example_ginisplits.cuh"

// Add the correct namespace or typedef if needed


int main() {

  std::string filename = "../generated_data_5.csv";
  auto dataset = Utils::flattenCSVColumnMajor<float>(filename);

  const int M               = 4;    // number of samples
  const int N               = 2;       // total number of features
  const int n_sampled_cols  = 2;       // Sample every feature
  const int num_nodes       = 1;       // two nodes in this example
  const int max_n_bins      = 4;
  const int max_depth       = 3; //not used
  const int min_samples_split = 2;
  const int max_leaves      = 4; //not used
  const uint64_t seed       = 2025ULL;
  const int treeid          = 0;
  const int colStart        = 0;
  const int num_classes     = 2; // number of classes for classification, binary for this example
  const int min_samples_leaf  = 1;
  ML::DT::GiniObjectiveFunction<float, int, int> objective(num_classes, min_samples_leaf);

  //
  // Host data arrays
  //
  std::vector<float>  h_data(M * N);
  h_data = {0.1, 0.4, 0.6, 0.9, 0.3, 0.2, 0.8, 0.5};
  // Fill data in column-major order
  // for (int col = 0; col < N; ++col) {
  //     for (int row = 0; row < M; ++row) {
  //         h_data[col * M + row] = (row * 1.0f + col) / (M + N);
  //     }
  // }
  std::vector<int> h_labels(M);
  for (int i = 0; i < M; ++i) {
      h_labels[i] = static_cast<int>(i % num_classes);
  }


GiniGainsBestSplits<float, int, int>(h_data,
                                     h_labels,
                                     M,    // number of samples
                                     N,       // total number of features
                                     n_sampled_cols,       // Sample every feature
                                     num_nodes,       // two nodes in this example
                                     max_n_bins,
                                     max_depth,//not used
                                     min_samples_split,
                                     max_leaves,//not used
                                     seed,
                                     treeid,
                                     colStart,
                                     num_classes, // number of classes for classification, binary for this example
                                     min_samples_leaf,
                                     objective);


  return 0;
}