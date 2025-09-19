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
#include "example_apply.cuh"
#include "example_ginisplits.cuh"
#include "utils.hpp"

int main (int argc, char* argv[]) {

    if (argc != 7) {
        std::cerr << "Usage: " << argv[0] << " <path> <selected_features_count> <num_proj> <max_n_bins> <col_gen_seed> <verbose>" << std::endl;
        return 1;
    }
    //First argument: File Path
    const std::string path = argv[1]; 
    //Second argument: Selected Features Count
    const int selected_features_count = std::atoi(argv[2]);// K number, selected number of features.
    //Third argument: Number of Projections
    const int num_proj = std::atoi(argv[3]);
    //Fourth argument: Verbose for debug information
    const int max_n_bins = std::atoi(argv[4]);
    const int col_gen_seed = std::atoi(argv[5]);


    bool verbose = false;
    if (std::string(argv[6]) == "true"){
        verbose = true;
    }

    //Load dataset
    auto dataset = Utils::flattenCSVColumnMajorWithLabels<float, int>(path);

    int num_rows = dataset.num_rows;
    int num_cols = dataset.num_cols;
    std::cout << std::endl;
    std::cout << "Dataset number of rows: " << num_rows << std::endl;
    std::cout << "Dataset number of columns: " << num_cols << std::endl;

    //Create flat dataset for CUDA, column major flattened 
    std::vector<float> flat_data = dataset.flattened;
    printf("First data: %lf\n", dataset.flattened[0]);

    std::vector<int> h_labels = dataset.labels;
    printf("First label: %d\n", dataset.labels[0]);

    Utils::RandomConfig cfg;
    cfg.minValue = 0;
    cfg.maxValue = num_cols - 1;
    cfg.count    = selected_features_count * num_proj; // total number of indices needed
    cfg.unique   = true;
    cfg.seed     = col_gen_seed;        // comment this line → fresh set every run

    auto numbers = Utils::generateRandom(cfg);

    std::cout << "Random picks:";
    for (int n : numbers) std::cout << ' ' << n;
    std::cout << '\n';

     ////////////////////////////////////////////////////////////ColAdd//////////////////////////////////////////
    //Create flat dataset for CUDA, column major flattened 
    double elapsed_ms = 0.0;
    int total_col_size = num_proj * selected_features_count;
    std::vector<float> GPU_Col_Add_values(num_rows * num_proj);
    std::vector<int> total_col_indices(total_col_size);

    //////////////////////////////////////////////////////debug
    // Copy generated random indices to total_col_indices
    std::copy(numbers.begin(), numbers.end(), total_col_indices.begin());
    //////////////////////////////////////////////////////debug
 
    float* d_col_add_projected = nullptr;
    ApplyProjectionColumnADD (flat_data.data(),
                              &d_col_add_projected,  // <-- receive device pointer
                              GPU_Col_Add_values.data(),
                              total_col_indices.data(),
                              num_rows,
                              num_cols,
                              num_proj,
                              selected_features_count,
                              elapsed_ms,
                              verbose);
    if (d_col_add_projected == nullptr) {
        std::cerr << "Error: d_col_add_projected is null." << std::endl;
        return 1;
    }
    float* h_buffer = new float[5];
    cudaMemcpy(h_buffer, d_col_add_projected, 5 * sizeof(float), cudaMemcpyDeviceToHost);
    // printf("d_col_add_projected values: ");
    // for (int i = 0; i < 5; ++i) {
    //     printf("%f, ", h_buffer[i]);
    // }
    ///////////////////////////////////////////////////////////ColAdddata////////////////////////////////////////
    if (verbose) {
        printf("Label is: ");
        for (int value : h_labels){  
            printf("%d, ", value);
        }
        printf("\n");

        printf("d_col_add_projected address: %p \n", (void*)d_col_add_projected);
        int N = /* number of elements in d_col_add_projected */ num_rows * num_proj;

        // Allocate host buffer
        float* h_buffer = new float[N];

        // Copy from device to host
        cudaMemcpy(h_buffer, d_col_add_projected, N * sizeof(float), cudaMemcpyDeviceToHost);

        // Print values
        printf("d_col_add_projected values: ");
        for (int i = 0; i < N; ++i) {
            printf("%f, ", h_buffer[i]);
        }
        printf("\n");
        std::cout << "Generated column indices: " << std::endl; 
        for (float value : total_col_indices) {
        std::cout << value << " "; 
        }
        std::cout << std::endl;
        printf("\n");
        // for (int i = 0; i < N; ++i) {
        //     printf("d_col_add_projected[%d] = %f\n", i, h_buffer[i]);
        // }
        printf("num_rows: %d\n", num_rows);
        printf("num_proj: %d\n", num_proj);

    // Clean up
    delete[] h_buffer;
    }

    std::cout << "Apply projection time elapsed: " << elapsed_ms << " ms " << std::endl;
    ////////////////////////////////////////////////////////////ColAdddata////////////////////////////////////////

    /////////////////////////////////////////////////GiniGainsBestSplits//////////////////////////////////////////

    const int M               = num_rows;    // num_rows
    const int N               = num_proj;       // num_proj
    const int n_sampled_cols  = num_proj;       // Sample every feature if < n_sampled_cols = num_proj;
    const int num_nodes       = 1;       // 1 node in this example
    const int max_depth       = 3; //not used
    const int min_samples_split = 2; //not usedd_col_add_projected
    const int max_leaves      = 4; //not used
    const uint64_t seed       = 2025ULL;
    const int treeid          = 0;
    const int colStart        = 0;
    const int num_classes     = 2; // number of classes for classification, binary for this example
    const int min_samples_leaf  = 0; 
    ML::DT::GiniObjectiveFunction<float, int, int> objective(num_classes, min_samples_leaf);
    
    double elapsed_gini_ms = 0.0;
    std::vector<ML::DT::Split<float, int>> h_splits(num_nodes);
    GiniGainsBestSplits<float, int, int>(d_col_add_projected, //replace with col add flattened data
                                        h_labels,
                                        M,
                                        N,       // total number of features/projections in this case for oblique
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
                                        objective,
                                        h_splits,
                                        elapsed_gini_ms,
                                        verbose);
   
    std::cout << "Gini + Histogram time elapsed: " << elapsed_gini_ms << " ms " << std::endl;
    std::cout << "cuML Total time elapsed: " << (elapsed_ms + elapsed_gini_ms) << " ms " << std::endl;
     for (int nid = 0; nid < num_nodes; ++nid) {
        auto &sp = h_splits[nid];
          //std::cout << "Best split for node " << nid << ":\n";
          std::cout << "  Projection: " << sp.colid << "\n";
          std::cout << "  Gain: " << sp.best_metric_val << "\n";
          std::cout << "  Threshold: " << sp.quesval << "\n";
          
    }
    ///////////////////////////GiniGainsBestSplits//////////////////////////////////////////
    return 0;
    }
