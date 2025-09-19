#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <iomanip>  // for std::setprecision

// Function to generate random float between min and max
float getRandomFloat(std::mt19937& gen, float min = 0.0f, float max = 100.0f) {
    std::uniform_real_distribution<float> dist(min, max);
    return dist(gen);
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <num_rows> <num_cols>" << std::endl;
        return 1;
    }
    const int num_rows = std::atoi(argv[1]);
    const int num_cols = std::atoi(argv[2]);

    std::string filename = "generated_data.csv";

    // Random number setup
    std::random_device rd;
    std::mt19937 gen(rd());

    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open file for writing.\n";
        return 1;
    }

    // Optional: write a header
    for (int c = 0; c < num_cols; ++c) {
        file << "Feature" << c;
        if (c != num_cols - 1) file << ",";
    }
    file << "\n";

    // Generate data
    for (int r = 0; r < num_rows; ++r) {
        for (int c = 0; c < num_cols; ++c) {
            float val = getRandomFloat(gen);
            file << std::fixed << std::setprecision(4) << val;
            if (c != num_cols - 1) file << ",";
        }
        file << "\n";
    }

    file.close();
    std::cout << "CSV file '" << filename << "' generated with "
              << num_rows << " rows and " << num_cols << " columns.\n";

    return 0;
}