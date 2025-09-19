#pragma once
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <type_traits>

namespace Utils {

    template <typename T>
    struct CSVData {
        std::vector<T> flattened;
        size_t num_rows;
        size_t num_cols;
    };

    template <typename T>
    CSVData<T> flattenCSVColumnMajor(const std::string& filename) {
        static_assert(std::is_arithmetic<T>::value, "CSVData type must be numeric");

        std::ifstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Failed to open file: " << filename << std::endl;
            return {{}, 0, 0};
        }

        std::vector<std::vector<T>> rows;
        std::string line;

        // Skip header
        std::getline(file, line);

        while (std::getline(file, line)) {
            std::stringstream ss(line);
            std::string cell;
            std::vector<T> row;

            while (std::getline(ss, cell, ',')) {
                try {
                    if constexpr (std::is_same<T, int>::value) {
                        row.push_back(std::stoi(cell));
                    } else if constexpr (std::is_same<T, float>::value) {
                        row.push_back(std::stof(cell));
                    } else if constexpr (std::is_same<T, double>::value) {
                        row.push_back(std::stod(cell));
                    } else {
                        row.push_back(static_cast<T>(std::stod(cell))); // fallback
                    }
                } catch (const std::invalid_argument& e) {
                    std::cerr << "Invalid value: " << cell << std::endl;
                    row.push_back(static_cast<T>(0)); // fallback
                }
            }

            rows.push_back(row);
        }

        file.close();

        if (rows.empty()) {
            std::cerr << "CSV is empty.\n";
            return {{}, 0, 0};
        }

        size_t num_rows = rows.size();
        size_t num_cols = rows[0].size();

        std::vector<T> flattened;
        flattened.reserve(num_rows * num_cols);

        for (size_t col = 0; col < num_cols; ++col) {
            for (size_t row = 0; row < num_rows; ++row) {
                flattened.push_back(rows[row][col]);
            }
        }

        return {flattened, num_rows, num_cols};
    }

} // namespace Utils
