import os
import sys
import numpy as np


def read_matrix(filename):
    with open(filename) as f:
        f.readline()
        data = [list(map(int, line.split())) for line in f]
    return np.array(data)


# Забираем размер, который передает C++
size = sys.argv[1]

base_dir = os.path.dirname(os.path.abspath(__file__))
matrix_dir = os.path.join(base_dir, "..", "matrix")

# Подставляем размер прямо в имена файлов
M1_path = os.path.join(matrix_dir, f"matrix_{size}_1.txt")
M2_path = os.path.join(matrix_dir, f"matrix_{size}_2.txt")
Mres_cpp_path = os.path.join(matrix_dir, f"result_{size}.txt")

M1 = read_matrix(M1_path)
M2 = read_matrix(M2_path)
Mres_cpp = read_matrix(Mres_cpp_path)

Mres_np = M1 @ M2

if np.array_equal(Mres_cpp, Mres_np):
    print(f"OK: results match for size {size}")
else:
    print(f"ERROR: results do not match for size {size}")
    # Debug info on English:
    print(f"--> M1 shape: {M1.shape}")
    print(f"--> M2 shape: {M2.shape}")
    print(f"--> C++ Result shape: {Mres_cpp.shape}")
    print(f"--> Expected NumPy shape: {Mres_np.shape}")
    
    # Print first few elements to see where the mismatch is
    if Mres_cpp.ndim > 1 and Mres_np.ndim > 1:
        print(f"--> First 3 elements C++: {Mres_cpp[0][:3]}")
        print(f"--> First 3 elements NumPy: {Mres_np[0][:3]}")