#include <iostream>
#include <fstream>
#include <vector>
#include <cstdlib>
#include <chrono>
#include <format>
#include <string>
#include <thread>
#include <limits>
#include "mpi.h"

using ll = long long;

// Безопасное чтение матрицы с очисткой потока от мусора в первой строке
std::vector<std::vector<ll>> read_matrix(const std::string& filename) {
    std::ifstream file(filename);

    if (!file.is_open()) {
        std::cerr << "Error: cannot open file " << filename << "\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    size_t rows = 0, cols = 0;
    if (!(file >> rows >> cols)) {
        std::cerr << "Error: invalid matrix dimensions in " << filename << "\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    if (rows <= 0 || cols <= 0) {
        std::cerr << "Error: invalid matrix size in " << filename << "\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Пропускаем остаток первой строки (символы переноса строк \r\n или пробелы)
    file.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

    std::vector<std::vector<ll>> matrix(rows, std::vector<ll>(cols));

    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            if (!(file >> matrix[i][j])) {
                std::cerr << "Error: invalid matrix format in " << filename << "\n";
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
        }
    }

    return matrix;
}

// Запись матрицы (вызывается строго на rank 0)
void write_matrix(const std::string& filename, const std::vector<std::vector<ll>>& M) {
    if (M.empty()) return;

    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: cannot open file for writing: " << filename << "\n";
        return;
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

// Параллельное умножение с использованием векторного сборщика MPI_Gatherv
std::vector<std::vector<ll>> multiply_matrix(const std::vector<std::vector<ll>>& M1, const std::vector<std::vector<ll>>& M2, size_t processes, size_t rank) {
    size_t M1_rows = M1.size();
    size_t M1_cols = M1[0].size();
    size_t M2_rows = M2.size();
    size_t M2_cols = M2[0].size();

    // Расчет распределения строк с учетом остатка (аналогично рабочему коду)
    size_t base_rows = M1_rows / processes;
    size_t rem_rows = M1_rows % processes;

    size_t start = rank * base_rows + (rank < rem_rows ? rank : rem_rows);
    size_t my_rows = base_rows + (rank < rem_rows ? 1 : 0);

    std::vector<std::vector<ll>> local_result(my_rows, std::vector<ll>(M2_cols, 0));

    // Локальное вычисление подматрицы
    for (size_t i = 0; i < my_rows; ++i) {
        size_t global_i = start + i;
        for (size_t j = 0; j < M2_cols; ++j) {
            ll sum = 0;
            for (size_t k = 0; k < M1_cols; ++k) {
                sum += M1[global_i][k] * M2[k][j];
            }
            local_result[i][j] = sum;
        }
    }

    // Линеаризация локального буфера перед отправкой в MPI
    std::vector<ll> local_flat(my_rows * M2_cols);
    size_t idx = 0;
    for (size_t i = 0; i < my_rows; i++) {
        for (size_t j = 0; j < M2_cols; j++) {
            local_flat[idx++] = local_result[i][j];
        }
    }

    std::vector<ll> full_flat;
    std::vector<int> recvcounts(processes, 0);
    std::vector<int> displs(processes, 0);

    // Подготовка служебных массивов для MPI_Gatherv (только на rank 0)
    if (rank == 0) {
        full_flat.resize(M1_rows * M2_cols);
        int current_displ = 0;
        for (size_t p = 0; p < processes; ++p) {
            size_t p_rows = base_rows + (p < rem_rows ? 1 : 0);
            recvcounts[p] = static_cast<int>(p_rows * M2_cols);
            displs[p] = current_displ;
            current_displ += recvcounts[p];
        }
    }

    // Собираем данные со всех процессов (даже если у них разное количество строк)
    MPI_Gatherv(
        local_flat.data(),
        static_cast<int>(my_rows * M2_cols),
        MPI_LONG_LONG,
        rank == 0 ? full_flat.data() : nullptr,
        recvcounts.data(),
        displs.data(),
        MPI_LONG_LONG,
        0,
        MPI_COMM_WORLD
    );

    std::vector<std::vector<ll>> result;
    if (rank == 0) {
        result.resize(M1_rows, std::vector<ll>(M2_cols));
        size_t id = 0;
        for (size_t i = 0; i < M1_rows; i++) {
            for (size_t j = 0; j < M2_cols; j++) {
                result[i][j] = full_flat[id++];
            }
        }
    }

    return result;
}

// Генератор матриц (вызывается строго на rank 0)
void generate_matrix(size_t rows, size_t cols, const std::string& output_filename, int min = -5, int max = 5) {
    std::string command = std::format("python python\\matrix_generator.py -r {} -c {} -m {} -M {} -o {}", rows, cols, min, max, output_filename);
    system(command.c_str());
}

// Тест принимает уникальные имена файлов для текущего размера матрицы
double run_single_test(size_t size, size_t processes, size_t rank) {
    // Пути, по которым C++ будет ЧИТАТЬ файлы матриц (от корня проекта)
    std::string f1 = std::format("matrix/matrix_{}_1.txt", size);
    std::string f2 = std::format("matrix/matrix_{}_2.txt", size);
    std::string f_res = std::format("matrix/result_{}.txt", size);

    auto m1 = read_matrix(f1);
    auto m2 = read_matrix(f2);

    auto start = std::chrono::high_resolution_clock::now();
    auto result = multiply_matrix(m1, m2, processes, rank);
    auto end = std::chrono::high_resolution_clock::now();

    if (rank == 0) {
        // Записываем результат работы C++
        write_matrix(f_res, result);
        
        // Передаем питону ТОЛЬКО чистый размер матрицы (например, 400)
        // Питон сам подставит его куда нужно и склеит правильные имена
        std::string verification_cmd = std::format("python python\\verification.py {}", size); 
        system(verification_cmd.c_str());
    }

    MPI_Barrier(MPI_COMM_WORLD); 

    return std::chrono::duration<double, std::milli>(end - start).count();
}
void log_result(size_t size, double avg_time, size_t processes) {
    std::ofstream log("src/measurements.txt", std::ios::app);
    log << "Size: " << size << " | Time: " << avg_time << " ms | Processes: " << processes << "\n";
}

void run_benchmark_for_size(size_t size, size_t repeats, size_t processes, size_t rank) {
    std::vector<double> times;
    
    // Переменные путей:
    // Для C++ (чтение) нужен полный путь от корня
    std::string cpp_f1 = std::format("matrix/matrix_{}_1.txt", size);
    std::string cpp_f2 = std::format("matrix/matrix_{}_2.txt", size);

    // Для Python передаем ТОЛЬКО чистое имя, так как он сам подставит папки внутри скрипта
    std::string py_f1 = std::format("matrix_{}_1.txt", size);
    std::string py_f2 = std::format("matrix_{}_2.txt", size);

    for (size_t i = 0; i < repeats; i++) {
        if (rank == 0) {
            // Передаем питону имена БЕЗ "matrix/"
            generate_matrix(size, size, py_f1);
            generate_matrix(size, size, py_f2);
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
        }
        
        MPI_Barrier(MPI_COMM_WORLD); 

        // C++ читает файлы по полному пути
        double time = run_single_test(size, processes, rank);
        times.push_back(time);
    }

    if (rank == 0) {
        double avg = 0.0;
        for (double t : times) avg += t;
        avg /= repeats;

        log_result(size, avg, processes);
        std::cout << "  " << size << "x" << size << " на " << processes << " процессах: " << avg << " ms\n";
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    
    std::vector<size_t> sizes = {200, 400, 800, 1200, 1600, 2000};
    const size_t tests_per_config = 3;

    int rank, processes;
    MPI_Comm_size(MPI_COMM_WORLD, &processes);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Только rank 0 очищает файл логов при старте программы
    if (rank == 0) {
        std::ofstream("src/measurements.txt", std::ios::trunc).close();
    }

    for (size_t size : sizes) {  
        run_benchmark_for_size(size, tests_per_config, processes, rank);
        if (rank == 0) {
            std::cout << "\n";
        }
    }

    MPI_Finalize();
    return 0;
}