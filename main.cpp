#include <iostream>
#include <fstream>
#include <vector>


std::vector<std::vector<int>> read_matrix(const std::string& filename, int rows, int cols) {
    std::ifstream file(filename);

    if (!file.is_open()) {
        std::cerr << "Error: cannot open file\n";
        exit(1);
    }

    std::vector<std::vector<int>> matrix(rows, std::vector<int>(cols));

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            if (!(file >> matrix[i][j])) {
                std::cerr << "Error: invalid matrix format\n";
                exit(1);
            }
        }
    }

    return matrix;
}


void write_matrix(const std::string& filename, const std::vector<std::vector<int>>& M) {
    if (M.empty()) {
        std::cerr << "Error: empty matrix\n";
        exit(1);
    }

    std::ofstream file(filename);

    if (!file.is_open()) {
        std::cerr << "Error: cannot open file for writing\n";
        exit(1);
    }

    file << M.size() << "\t" << M[0].size() << "\n";

    for (const auto& row : M) {
        for (size_t j = 0; j < row.size(); ++j) {
            file << row[j];
            if (j + 1 != row.size()) file << " ";
        }
        file << "\n";
    }
}


std::vector<std::vector<int>> multiply_matrix(std::vector<std::vector<int>> M1, const std::vector<std::vector<int>> M2) {
    if (M1.empty() || M2.empty()) {
        std::cerr << "Error: empty matrix\n";
        exit(1);
    }

    int M1_rows = M1.size();
    int M1_cols = M1[0].size();
    int M2_rows = M2.size();
    int M2_cols = M2[0].size();

    if (M1_cols != M2_rows) {
        std::cerr << "Error: matrices cannot be multiplied\n";
        exit(1);
    }

    std::vector<std::vector<int>> result(M1_rows, std::vector<int>(M2_cols, 0));

    for (int i = 0; i < M1_rows; ++i) {
        for (int j = 0; j < M2_cols; ++j) {
            for (int k = 0; k < M1_cols; ++k) {
                result[i][j] += M1[i][k] * M2[k][j];
            }
        }
    }

    return result;
}


int main(){

    return 0;
}