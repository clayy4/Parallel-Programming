#include <iostream>
#include <fstream>
#include <vector>
#include <cstdlib>
#include <chrono>
#include <format>
#include <string>
#include <omp.h>

using ll = long long;

std::vector<std::vector<ll>> read_matrix(const std::string& filename) {
    std::ifstream file(filename);

    if (!file.is_open()) {
        std::cerr << "Error: cannot open file\n";
        exit(1);
    }

    size_t rows, cols;
    file >> rows >> cols;

    if (rows <= 0 || cols <= 0) {
        std::cerr << "Error: invalid matrix size\n";
        exit(1);
    }

    std::vector<std::vector<ll>> matrix(rows, std::vector<ll>(cols));

    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            if (!(file >> matrix[i][j])) {
                std::cerr << "Error: invalid matrix format\n";
                exit(1);
            }
        }
    }

    return matrix;
}


void write_matrix(const std::string& filename, const std::vector<std::vector<ll>>& M) {
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


std::vector<std::vector<ll>> multiply_matrix(const std::vector<std::vector<ll>>& M1, const std::vector<std::vector<ll>>& M2) {
    if (M1.empty() || M2.empty()) {
        std::cerr << "Error: empty matrix\n";
        exit(1);
    }

    size_t M1_rows = M1.size();
    size_t M1_cols = M1[0].size();
    size_t M2_rows = M2.size();
    size_t M2_cols = M2[0].size();

    if (M1_cols != M2_rows) {
        std::cerr << "Error: matrices cannot be multiplied\n";
        exit(1);
    }

    std::vector<std::vector<ll>> result(M1_rows, std::vector<ll>(M2_cols, 0));

    #pragma omp parallel for collapse(2) schedule(static)
    for (size_t i = 0; i < M1_rows; ++i) {
        for (size_t j = 0; j < M2_cols; ++j) {
            ll sum = 0;
            for (size_t k = 0; k < M1_cols; ++k) {
                sum += M1[i][k] * M2[k][j];
            }
            result[i][j] = sum;
        }
    }

    return result;
}


bool processing(std::vector<size_t>& threads, std::vector<std::vector<std::string>>& data, size_t test_num){
    for(const size_t& thread: threads) {
        omp_set_num_threads(thread);
        for(const auto& d: data){
            std::string rows, cols, min, max, path_m1, path_m2;
            rows = d[0];
            cols = d[1];
            min = d[2]; 
            max = d[3];
            path_m1 = d[4];
            path_m2 = d[5];

            std::vector<double> timer(test_num, 0.0);
            for(size_t i = 0;  i < test_num; i++){
                    std::string command1 = std::format("python python\\matrix_generator.py -r {} -c {} -m {} -M {} -o {}", rows, cols, min, max, path_m1);
                    std::string command2 = std::format("python python\\matrix_generator.py -r {} -c {} -m {} -M {} -o {}", rows, cols, min, max, path_m2);
                    system(command1.c_str());
                    system(command2.c_str());

                    std::vector<std::vector<ll>> matrix1 = read_matrix("matrix/matrix1.txt");
                    std::vector<std::vector<ll>> matrix2 = read_matrix("matrix/matrix2.txt");

                    auto start = std::chrono::high_resolution_clock::now();

                    std::vector<std::vector<ll>> result = multiply_matrix(matrix1, matrix2);

                    auto end = std::chrono::high_resolution_clock::now();

                    std::chrono::duration<double, std::milli> duration = end - start;
                    timer[i] = duration.count();

                    write_matrix("matrix/result.txt", result);

                    system("python python\\verification.py");
            }
            
            double avg = 0.0;
            for(double time: timer){
                avg+=time;
            }
            avg/=3.0;

            
            std::ofstream file("src/measurements.txt", std::ios::app);

            file << "Size: " << rows
                << " | Time: " << avg << " ms"
                << " | Threads: " << thread
                << "\n";

        }
    }
    return true;
}


int main() {
    std::vector<size_t> threads = {1, 2, 4, 8};
    std::vector<std::vector<std::string>> data = {
        {"200",  "200",  "-10000", "10000", "matrix1.txt", "matrix2.txt"},
        {"400",  "400",  "-10000", "10000", "matrix1.txt", "matrix2.txt"},
        {"800",  "800",  "-10000", "10000", "matrix1.txt", "matrix2.txt"},
        {"1200", "1200", "-10000", "10000", "matrix1.txt", "matrix2.txt"},
        {"1600", "1600", "-10000", "10000", "matrix1.txt", "matrix2.txt"},
        {"2000", "2000", "-10000", "10000", "matrix1.txt", "matrix2.txt"}
    };
    size_t number_of_tests = 3;

    processing(threads, data, number_of_tests);
    

    return 0;
}