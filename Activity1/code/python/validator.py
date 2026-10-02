import csv
import os
import numpy as np
from matrix_mult import multiply_matrices

def load_matrix(filename):
    with open(filename, 'r') as f:
        return [[float(val) for val in row] for row in csv.reader(f)]

def main():
    n = 10
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.abspath(os.path.join(current_dir, '../../data'))
    
    path_A = os.path.join(data_dir, f'matrix_A_{n}.csv')
    path_B = os.path.join(data_dir, f'matrix_B_{n}.csv')
    path_java = os.path.join(data_dir, f'result_Java_{n}.csv')
    path_c = os.path.join(data_dir, f'result_C_{n}.csv')

    if not os.path.exists(path_A) or not os.path.exists(path_B):
        print("Error: No se encuentran las matrices de entrada. Ejecuta primero generate_inputs.py")
        return

    A_list = load_matrix(path_A)
    B_list = load_matrix(path_B)
    
    A_np = np.array(A_list)
    B_np = np.array(B_list)

    C_ref = np.dot(A_np, B_np)

    print(f"--- VALIDATION OF CORRECTNESS (n = {n}) ---")

    C_py = multiply_matrices(A_list, B_list, n)
    is_py_correct = np.allclose(C_ref, np.array(C_py), atol=1e-6)
    print(f" * Python : {'CORRECT' if is_py_correct else 'INCORRECT'}")

    if os.path.exists(path_java):
        C_java = load_matrix(path_java)
        is_java_correct = np.allclose(C_ref, np.array(C_java), atol=1e-6)
        print(f" * Java: {'CORRECT' if is_java_correct else 'INCORRECT'}")
    else:
        print(f" * Java: FILE NOT FOUND ({path_java})")

    if os.path.exists(path_c):
        C_c = load_matrix(path_c)
        is_c_correct = np.allclose(C_ref, np.array(C_c), atol=1e-6)
        print(f" * C: {'CORRECT' if is_c_correct else 'INCORRECT'}")
    else:
        print(f" * C: FILE NOT FOUND ({path_c})")



if __name__ == "__main__":
    main()