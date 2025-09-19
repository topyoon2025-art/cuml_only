// Online C++ compiler to run C++ program online
#include <iostream>
#include <vector>
#include <cstdlib>   // for rand()
#include <ctime>     // for time()
#include <random>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <iomanip>
#include <vector>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cublas_v2.h>
#include <cstdint>
#include <chrono>


//warm-up kernel
__global__ void warmup() {}

__global__ void RandomColumnGenerationKernel(int* total_col_indices,
                                             int* shuffle_buffer, 
                                             int num_cols, 
                                             int selected_features_count, 
                                             int num_proj, 
                                             unsigned long long seed) {

    int thread_id_in_block = threadIdx.y * blockDim.x + threadIdx.x;
    int threads_per_block = blockDim.x * blockDim.y;
    int block_id = blockIdx.y * gridDim.x + blockIdx.x;
    int proj_id = block_id * threads_per_block + thread_id_in_block;

    if (proj_id >= num_proj) return;

    // Initialize RNG
    curandState state;
    curand_init(seed, proj_id, 0, &state);

    // Each thread gets its own slice of shared memory
    int* local_indices = shuffle_buffer + proj_id * num_cols;

    // Fill array 0..(num_cols - 1)
    for (int i = 0; i < num_cols; ++i) {
        local_indices[i] = i;
    }

    __syncthreads(); // sync before shuffle if threads cooperate

    // Shuffle the last selected_features_count elements
    for (int i = num_cols - 1; i >= num_cols - selected_features_count; --i) {
        int j = curand(&state) % (i + 1);
        int tmp = local_indices[i];
        local_indices[i] = local_indices[j];
        local_indices[j] = tmp;
    }

    // Write results to global memory
    int offset = proj_id * selected_features_count;
    for (int i = 0; i < selected_features_count; ++i) {
        total_col_indices[offset + i] = local_indices[num_cols - selected_features_count + i];
        //printf("proj %d: col[%d] = %d\n", proj_id, i, total_col_indices[offset + i]);
    }
}


__global__ void ColumnAddProjectionKernel(
  // [num_total_rows * num_cols]
  // [num_rows * num_cols]
  // [num_cols * num_proj]
  // [num_rows * num_proj]
    const float* __restrict__ dataset,            
    const int* __restrict__ flat_col_data,      
    float* projected,               
    int num_rows,
    int num_cols,
    int num_proj, 
    int selected_features_count) 
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;  // row index
    int col = blockIdx.x * blockDim.x + threadIdx.x;  // projection column index
    if (row < num_rows && col < num_proj) {
        float sum = 0.0f;
        for (int i = 0; i < selected_features_count; ++i ) 
        {
          int feature_idx = flat_col_data[col * selected_features_count + i];
            // Column-major access
          sum += dataset[feature_idx * num_rows + row];
        }
        //projected[row * num_proj + col] = sum;
        projected[col * num_rows + row] = sum;

    }
}


void ApplyProjectionColumnADD (const float* flat_data,
                               //float* GPU_Col_Add_values,
                               float** d_col_add_projected_out,  // <-- new
                               const int num_rows,
                               const int num_cols,
                               const int num_proj,
                               const int selected_features_count,
                               double& elapsed_ms,
                               bool verbose
                              ){

// Warm-up launch, The first kernel won’t be artificially inflated by setup costs
  warmup<<<1, 1>>>();
  cudaDeviceSynchronize();

  int total_dataset_size = num_rows * num_cols;
  int total_col_dataset_size = selected_features_count * num_proj;
  int result_size = num_rows * num_proj;
  std::cout << std::endl;

///////////////////////////////////////////////////Debug/////////////////////////////////////
  if (verbose) {
    std::cout << "Passed col add data from main to GPU function: " << std::endl;
    std::cout << "rows: " << num_rows << std::endl;
    std::cout << "cols: " << num_cols << std::endl;
    std::cout << "proj: " << num_proj << std::endl;
  }
  ///////////////////////////////////////////////////Debug/////////////////////////////////////

  //Allocate device memory
  float *d_flat_data = nullptr;
  float *d_col_add_projected = nullptr;
  int *d_flat_col_data = nullptr;
  int* d_shuffle_buffer = nullptr;

  cudaMalloc((void **)&d_flat_data, total_dataset_size * sizeof(float));
  cudaMalloc((void **)&d_flat_col_data, total_col_dataset_size * sizeof(int));                              
  cudaMalloc((void **)&d_col_add_projected, result_size * sizeof(float));
  cudaMalloc((void**)&d_shuffle_buffer, num_proj * num_cols * sizeof(int));
  cudaMemset(d_flat_data, 0, total_dataset_size * sizeof(float)); // Always zero/init before use
  cudaMemset(d_flat_col_data, 0, total_col_dataset_size * sizeof(int)); // Always zero/init before use
  cudaMemset(d_col_add_projected, 0, result_size * sizeof(float)); // Always zero/init before use
  cudaMemset(d_shuffle_buffer, 0, num_proj * num_cols * sizeof(int)); // Always zero/init before use


  //Copy dataset to device
  cudaMemcpy(d_flat_data, flat_data, total_dataset_size * sizeof(float), cudaMemcpyHostToDevice);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
      std::cerr << "CUDA ERROR: " << cudaGetErrorString(err) << std::endl;
  }

  // Launch CUDA kernel
  int block_size_x = 16;
  int block_size_y = 16;
  //int threads_per_block = block_size_x * block_size_y;
  dim3 block_size(block_size_x, block_size_y); 
  dim3 grid_size((num_proj + 15) / 16, (num_rows + 15) / 16); 

  unsigned long long seed = static_cast<unsigned long long>(time(NULL));
  RandomColumnGenerationKernel<<<grid_size, block_size>>>(d_flat_col_data,
                                                d_shuffle_buffer,
                                                num_cols, 
                                                selected_features_count,
                                                num_proj, 
                                                seed);
  cudaDeviceSynchronize();  // Ensure kernel finishes

  //Print comlumn data
  if (verbose) {
    std::vector<int> h_flat_col_data(num_proj * selected_features_count);
    cudaMemcpy(h_flat_col_data.data(), d_flat_col_data, h_flat_col_data.size() * sizeof(int), cudaMemcpyDeviceToHost);
    std::cout << "Generated column indices for projection:" << std::endl;
    for (int value: h_flat_col_data) {
      std::cout << value << " ";
    }
    std::cout << std::endl;
  }

  auto startA = std::chrono::high_resolution_clock::now();
  ColumnAddProjectionKernel<<<grid_size, block_size>>>(d_flat_data,
                                                       d_flat_col_data,   
                                                       d_col_add_projected,
                                                       num_rows, 
                                                       num_cols,
                                                       num_proj,
                                                       selected_features_count);
  cudaDeviceSynchronize();  // Ensure kernel finishes
  auto endA = std::chrono::high_resolution_clock::now();
  elapsed_ms = std::chrono::duration<double, std::milli>(endA - startA).count();

  *d_col_add_projected_out = d_col_add_projected;  // <-- pass it back

  //cudaMemcpy(GPU_Col_Add_values, d_col_add_projected, num_rows * num_proj * sizeof(float), cudaMemcpyDeviceToHost);

  cudaPeekAtLastError();
  // Free device memory
  cudaFree(d_flat_data);
  cudaFree(d_flat_col_data);
  // cudaFree(d_col_add_projected);
  cudaFree(d_shuffle_buffer);
}