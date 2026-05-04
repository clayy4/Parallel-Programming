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


void generate_matrix(size_t rows, size_t cols, const std::string& output_filename, int min=-10000, int max=10000) {
    std::string command = std::format("python python\\matrix_generator.py -r {} -c {} -m {} -M {} -o {}", rows, cols, min, max, output_filename);
    system(command.c_str());
}

double run_single_test() {
    auto m1 = read_matrix("matrix/matrix1.txt");
    auto m2 = read_matrix("matrix/matrix2.txt");

    auto start = std::chrono::high_resolution_clock::now();
    auto result = multiply_matrix(m1, m2);
    auto end = std::chrono::high_resolution_clock::now();

    write_matrix("matrix/result.txt", result);
    system("python python\\verification.py");

    return std::chrono::duration<double, std::milli>(end - start).count();
}


void log_result(size_t size, double avg_time, size_t threads) {
    std::ofstream log("src/measurements.txt", std::ios::app);
    log << "Size: " << size << " | Time: " << avg_time << " ms | Threads: " << threads << "\n";
}


void run_benchmark_for_size(size_t size, size_t threads, size_t repeats) {
    omp_set_num_threads(threads);

    std::vector<double> times;
    for (size_t i = 0; i < repeats; i++) {
        generate_matrix(size, size, "matrix1.txt");
        generate_matrix(size, size, "matrix2.txt");

        double time = run_single_test();
        times.push_back(time);
    }

    double avg = 0.0;
    for (double t : times) avg += t;
    avg /= repeats;

    log_result(size, avg, threads);
    std::cout << "  " << threads << " threads: " << avg << " ms\n";
}


int main() {
    std::vector<size_t> threads_list = {1, 2, 4, 8};
    std::vector<size_t> sizes = {200, 400, 800, 1200, 1600, 2000};
    const size_t tests_per_config = 3;

    std::ofstream("src/measurements.txt", std::ios::trunc).close();

    for (size_t size : sizes) {        
        for (size_t threads : threads_list) {
            run_benchmark_for_size(size, threads, tests_per_config);
        }
        std::cout << "\n";
    }
    return 0;
}