import os
import numpy as np


def read_matrix(filename):
    with open(filename) as f:
        f.readline()
        data = [list(map(int, line.split())) for line in f]
    return np.array(data)


base_dir = os.path.dirname(os.path.abspath(__file__))
matrix_dir = os.path.join(base_dir, "..", "matrix")

M1_path = os.path.join(matrix_dir, "matrix1.txt")
M2_path = os.path.join(matrix_dir, "matrix2.txt")
Mres_cpp_path = os.path.join(matrix_dir, "result.txt")

M1 = read_matrix(M1_path)
M2 = read_matrix(M2_path)
Mres_cpp = read_matrix(Mres_cpp_path)

Mres_np = M1 @ M2

if np.array_equal(Mres_cpp, Mres_np):
    print("OK: results match")
else:
    print("ERROR: results do not match")