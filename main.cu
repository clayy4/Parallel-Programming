#include <iostream>
#include <fstream>
#include <vector>
#include <cstdlib>
#include <chrono>
#include <format>
#include <string>
#include <cuda_runtime.h>

using ll = long long;

#define CUDA_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char* file, int line) {
    if (code != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(code)
                  << " | " << file << ":" << line << "\n";
        exit(code);
    }
}

// ===================== IO =====================

std::vector<std::vector<ll>> read_matrix(const std::string& filename) {
    std::ifstream file(filename);

    if (!file.is_open()) {
        std::cerr << "Error: cannot open file\n";
        exit(1);
    }

    size_t rows, cols;
    file >> rows >> cols;

    std::vector<std::vector<ll>> matrix(rows, std::vector<ll>(cols));

    for (size_t i = 0; i < rows; ++i)
        for (size_t j = 0; j < cols; ++j)
            file >> matrix[i][j];

    return matrix;
}

void write_matrix(const std::string& filename, const std::vector<std::vector<ll>>& M) {
    std::ofstream file(filename);

    file << M.size() << " " << M[0].size() << "\n";

    for (const auto& row : M) {
        for (size_t j = 0; j < row.size(); ++j) {
            file << row[j] << (j + 1 == row.size() ? "" : " ");
        }
        file << "\n";
    }
}

// ===================== CUDA KERNEL =====================

__global__ void multiply_matrix_kernel(
    const ll* A,
    const ll* B,
    ll* C,
    int A_rows,
    int A_cols,
    int B_cols
) {
    // Динамическая разделяемая память для двух тайлов
    extern __shared__ ll shared[];

    const int TILE = blockDim.x; // Предполагаем blockDim.x == blockDim.y == tile_size

    ll* tileA = shared;
    ll* tileB = shared + TILE * TILE;

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    // Глобальные индексы элемента в результирующей матрице C
    const int row = blockIdx.y * TILE + ty;
    const int col = blockIdx.x * TILE + tx;

    ll sum = 0;

    // Итерация по тайлам вдоль общей границы матриц (A_cols / B_rows)
    for (int t = 0; t < (A_cols + TILE - 1) / TILE; ++t) {

        // Каждый поток загружает один элемент в tileA и один в tileB
        int aCol = t * TILE + tx;
        int bRow = t * TILE + ty;

        // Загрузка в тайл А (строка 'row', столбец 'aCol')
        if (row < A_rows && aCol < A_cols) {
            tileA[ty * TILE + tx] = A[row * A_cols + aCol];
        } else {
            tileA[ty * TILE + tx] = 0;
        }

        // Загрузка в тайл B (строка 'bRow', столбец 'col')
        if (bRow < A_cols && col < B_cols) {
            tileB[ty * TILE + tx] = B[bRow * B_cols + col];
        } else {
            tileB[ty * TILE + tx] = 0;
        }

        // Синхронизация: ждем пока весь блок заполнит тайлы
        __syncthreads();

        // Перемножение элементов текущего тайла
        for (int k = 0; k < TILE; ++k) {
            sum += tileA[ty * TILE + k] * tileB[k * TILE + tx];
        }

        // Синхронизация перед следующей итерацией (чтобы не перезаписать данные раньше времени)
        __syncthreads();
    }

    // Запись результата в глобальную память
    if (row < A_rows && col < B_cols) {
        C[row * B_cols + col] = sum;
    }
}

// ===================== CUDA MULTIPLY =====================

double multiply_matrix(
    const std::vector<std::vector<ll>>& M1,
    const std::vector<std::vector<ll>>& M2,
    int tile_size
) {
    int A_rows = static_cast<int>(M1.size());
    int A_cols = static_cast<int>(M1[0].size());
    int B_cols = static_cast<int>(M2[0].size());

    std::vector<ll> h_A(A_rows * A_cols);
    std::vector<ll> h_B(A_cols * B_cols);
    std::vector<ll> h_C(A_rows * B_cols, 0);

    for (int i = 0; i < A_rows; ++i)
        for (int j = 0; j < A_cols; ++j)
            h_A[i * A_cols + j] = M1[i][j];

    for (int i = 0; i < A_cols; ++i)
        for (int j = 0; j < B_cols; ++j)
            h_B[i * B_cols + j] = M2[i][j];

    ll *d_A, *d_B, *d_C;

    CUDA_CHECK(cudaMalloc(&d_A, h_A.size() * sizeof(ll)));
    CUDA_CHECK(cudaMalloc(&d_B, h_B.size() * sizeof(ll)));
    CUDA_CHECK(cudaMalloc(&d_C, h_C.size() * sizeof(ll)));

    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), h_A.size() * sizeof(ll), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), h_B.size() * sizeof(ll), cudaMemcpyHostToDevice));

    dim3 block(tile_size, tile_size);
    dim3 grid(
        (B_cols + tile_size - 1) / tile_size,
        (A_rows + tile_size - 1) / tile_size
    );

    size_t shared_mem = 2 * tile_size * tile_size * sizeof(ll);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    multiply_matrix_kernel<<<grid, block, shared_mem>>>(d_A, d_B, d_C, A_rows, A_cols, B_cols);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);

    CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, h_C.size() * sizeof(ll), cudaMemcpyDeviceToHost));

    // ================== ИСПРАВЛЕННЫЙ БЛОК: ЗАПИСЬ РЕЗУЛЬТАТА ==================
    // 1. Конвертируем одномерный вектор h_C обратно в двумерный формат
    std::vector<std::vector<ll>> M_res(A_rows, std::vector<ll>(B_cols));
    for (int i = 0; i < A_rows; ++i) {
        for (int j = 0; j < B_cols; ++j) {
            M_res[i][j] = h_C[i * B_cols + j];
        }
    }

    // 2. Записываем полученную матрицу в файл для верификации Python-скриптом
    write_matrix("matrix/result.txt", M_res);
    // =========================================================================

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    return static_cast<double>(ms);
}

// ===================== BENCHMARK =====================

void generate_matrix(size_t rows, size_t cols, const std::string& output_filename) {
    std::string cmd = std::format(
        "python python\\matrix_generator.py -r {} -c {} -o {}",
        rows, cols, output_filename
    );

    system(cmd.c_str());
}

double run_single_test(int tile_size) {

    auto m1 = read_matrix("matrix/matrix1.txt");
    auto m2 = read_matrix("matrix/matrix2.txt");

    double time = multiply_matrix(m1, m2, tile_size);

    system("python python\\verification.py");

    return time;
}

void run_benchmark_for_size(size_t size, int tile, int repeats) {

    std::vector<double> times;

    for (int i = 0; i < repeats; ++i) {
        generate_matrix(size, size, "matrix1.txt");
        generate_matrix(size, size, "matrix2.txt");

        times.push_back(run_single_test(tile));
    }

    double avg = 0;
    for (double t : times) avg += t;
    avg /= repeats;

    std::cout << "  Tile " << tile << "x" << tile
              << ": " << avg << " ms\n";
}

// ===================== MAIN =====================

int main() {

    std::vector<int> tiles = {8, 16, 32};
    std::vector<int> sizes = {200, 400, 800, 1200, 1600, 2000};

    for (int size : sizes) {

        std::cout << "\nMatrix size: " << size << "x" << size << "\n";

        for (int tile : tiles)
            run_benchmark_for_size(size, tile, 3);
    }

    return 0;
}