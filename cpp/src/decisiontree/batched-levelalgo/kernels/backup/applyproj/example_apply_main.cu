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
#include "utils.hpp"
#include "example_apply.cuh"

int main (int argc, char* argv[]) {

    if (argc != 5) {
        std::cerr << "Usage: " << argv[0] << " <path> <selected_features_count> <num_proj> <verbose>" << std::endl;
        return 1;
    }
    //First argument: File Path
    const std::string path = argv[1]; 
    //Second argument: Selected Features Count
    const int selected_features_count = std::atoi(argv[2]);// K number, selected number of features.
    //Third argument: Number of Projections
    const int num_proj = std::atoi(argv[3]);
    //Fourth argument: Verbose for debug information
    bool verbose = false;
    if (std::string(argv[4]) == "true"){
        verbose = true;
    }

    //Set float precision
    std::cout << std::setprecision(4);

    //Load dataset
    auto dataset = Utils::flattenCSVColumnMajor<float>(path);

    int num_rows = dataset.num_rows;
    int num_cols = dataset.num_cols;
    std::cout << std::endl;
    std::cout << "Dataset number of rows: " << num_rows << std::endl;
    std::cout << "Dataset number of columns: " << num_cols << std::endl;

    //Create flat dataset for CUDA, column major flattened 
    std::vector<float> flat_data = dataset.flattened;

    ////////////////////////////////////////////////////////////ColAdd//////////////////////////////////////////
    double elapsed_ms = 0.0;
    std::vector<float> GPU_Col_Add_values(num_rows * num_proj);
    ApplyProjectionColumnADD (flat_data.data(),
                              GPU_Col_Add_values.data(),
                              num_rows,
                              num_cols,
                              num_proj,
                              selected_features_count,
                              elapsed_ms,
                              verbose);
    
    ////////////////////////////////////////////////////////////ColAdddata////////////////////////////////////////
    if (verbose) {
        std::cout << "GPU Col Add Values: " << std::endl; 
        for (float value : GPU_Col_Add_values) {
        std::cout << value << " "; 
        }
        std::cout << std::endl;
    }
    std::cout << "GPU Col Add time elapsed: " << elapsed_ms << " ms " << std::endl;
    std::cout << std::endl;
    ////////////////////////////////////////////////////////////ColAdddata////////////////////////////////////////




    
}
