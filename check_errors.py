import numpy as np
import os

# Reads the solution matrix from the file it is stored in
def read_solution_matrix(filepath):
    matrix = []
    in_solution = False

    # Open the file
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()

            if line.startswith("Solution:"):
                in_solution = True
                continue

            if in_solution:
                if not line.startswith("|"):
                    break  # End of solution block

                # Remove | characters and split numbers
                values = line.strip("|").split()
                matrix.append([float(v) for v in values])

    return np.array(matrix)

# Create the reference matrices (serial solutions)
reference_file_1 = "./Results/serial/serial_1.txt"
serial_1 = read_solution_matrix(reference_file_1)
reference_file_2 = "./Results/serial/serial_2.txt"
serial_2 = read_solution_matrix(reference_file_2)
reference_file_3 = "./Results/serial/serial_3.txt"
serial_3 = read_solution_matrix(reference_file_3)

ref = [serial_1, serial_2, serial_3]

# List the directories for the different algorithms where the files are stored
row_dirs = ["./Results/row-wise/1/", "./Results/row-wise/2/", "./Results/row-wise/3/"]
col_dirs = ["./Results/column-wise/1/", "./Results/column-wise/2/", "./Results/column-wise/3/"]
nnz_dirs = ["./Results/nnz-wise/1/", "./Results/nnz-wise/2/", "./Results/nnz-wise/3/"]
PETSc_dirs = ["./Results/PETSc/1/", "./Results/PETSc/2/", "./Results/PETSc/3/"]
dirs = [row_dirs, col_dirs, nnz_dirs, PETSc_dirs]

# Check for every file in all the listed directories if the solution matrix is equal to the corresponding serial solution
for dir in range(len(dirs)):
    for i in range(len(dirs[dir])):
        for root, _, files in os.walk(dirs[dir][i]):
            for name in files:
                if not name.endswith(".txt"):
                    continue

                filepath = os.path.join(root, name)

                try:
                    matrix = read_solution_matrix(filepath)
                except Exception as e:
                    print(f"Could not read {filepath}: {e}")
                    continue

                if matrix.shape != ref[i].shape:
                    print(f"❌ Shape mismatch in {filepath}")
                    continue

                if np.allclose(matrix, ref[i], atol=1e-3):  # Tolerance of 0.001
                    print(f"✅ Match: {filepath}")
                else:
                    print(f"❌ Different values: {filepath}")