# Intel MPI Parallel Sparse-Matrix Fat-Vector Multiplication

This repository includes the code to parallelise a sparse matrix and fat vector multiplication using a distributed memory architecture and Intel MPI as the parallelisation tool.

The fat vector joins several vectors into one matrix to reduce the number of matrix and vector multiplications, computing only a single matrix multiplication to test the data with different parameters.

Regarding the parallelisation methods, 3 were designed, apart from the serial approach. For their description, for illusttration purposes, 3X3 matrix will be used as the sparse matrix and a 3X2 matrix as the fat vector, but these dimensions differ to the ones used for the computations. To refer to the real dimensions in a general way, the sparse matrix is a mXn matrix and the fat vector a nXk matrix. Additionally, the number of processes will be illustrated as 3, whereas this number will vary, and it will be referenced as npes.

## Serial algorithm

First, the problem was approached serially.

The serial solution for this problem is the regular matrix multiplication algorithm: the ith column of the sparse matrix is multiplied by the jth column of the fat vector, and the result of that multiplication will be the element (i, j) of the solution fat vector.

<img width="1024" height="231" alt="image" src="https://github.com/user-attachments/assets/6fa5260b-03a1-4915-b73b-e12ff5a71141" />

The previous image shows how the serial algorithm works to obtain the first element of the solution vector, where $u_{1,1} = M_{1,1} \cdot v_{1,1} + M_{1,2} \cdot v_{2,1} + M_{1,3} \cdot v_{3,1}$. This serial algorithm would go on, first computing all the elements from the first row of the solution fat vector and then continuing in the same way for the subsequent rows. 

Additionally, as it is a serial algorithm, all the computation would be done by the same CPU core.

Once the serial algorithm was determined, its parallelisation was designed. Three different parallelisation methods were considered for this problem to compare their performance and determine the parallel algorithm that best suits this computation. The programmes are called: row-wise, column-wise and nnz-wise, where nnz stands for number of non-zero elements.

## Row-wise parallel algorithm

As the name states, for the row-wise parallel algorithm, the different rows of the sparse matrix are distributed among the available processes. All the elements of each row are sent to a specific CPU core to compute its k corresponding results, which means that the whole fat vector will be also sent to each processor.

<img width="1181" height="661" alt="image" src="https://github.com/user-attachments/assets/7f74f3a1-c988-418e-a0a2-0552d6c97eb2" />

The previous image depicts how the row-wise parallel algorithm would work with 3 processes and the example matrices. Process 1 gets the first row of the sparse matrix, process 2 gets the second one and process 3 the third one, and the three of them get the full fat vector. Then, each process multiplies their local row by the fat vector, obtaining all the results for that row, and finally, all the local results from the different processes are gathered into the final solution fat vector.

With the real dimensions, to balance the load, each process receives m/npes rows, receiving m%npes (the remainder of m/npes) processes one additional row.

This means that, locally, each process has a set of rows from the sparse matrix, stored as an (m/npes)Xn or ((m/npes)+1)Xn matrix, and the fat vector, and will generate an ((m/npes)*n)Xk or (((m/npes)+1)*n)Xk matrix to store the local results.

Regarding the communication cost, it is expected to be on the order of $m \cdot npes + n \cdot k$, and the local computation cost will be $O \left (\frac{m \cdot n}{npes} \right)$.

## Column-wise parallel algorithm

As the name states, for the column-wise parallel algorithm, the different columns of the sparse matrix are distributed among the available processes. All the elements of each column are sent to a specific CPU core to compute its k corresponding results, which means that only the relevant rows from the fat vector will be also sent to each processor.

<img width="1181" height="658" alt="image" src="https://github.com/user-attachments/assets/db3c1ca3-2977-4c11-aa50-e82153e051bf" />

The previous image depicts how the column-wise parallel algorithm would work with 3 processes and the example matrices. Process 1 gets the first column of the sparse matrix and the first row of the fat vector, process 2 gets the second column from the matrix and the second row from the fat vector, and process 3 the third column from the matrix and the third row from the fat vector. Then, each process multiplies their local column by each element from the associated row from the fat vector, obtaining a partial result of the solution fat vector. Finally, all the local results from the different processes are gathered and summed, obtaining thus the final solution fat vector.

With the real dimensions, to balance the load, each process receives n/npes columns, receiving n%npes (the remainder of n/npes) processes one additional column.

This means that, locally, each process has a set of columns from the sparse matrix, stored as an mX(n/npes) or mX((n/npes)+1) matrix, and a set of rows from the fat vector, stored as an (n/npes)Xk or ((n/npes)+1)Xk matrix, and will generate an mXk matrix to store the local results.

Regarding the communication cost, it is expected to be on the order of $\frac {m \cdot n}{npes} + \frac {n \cdot k}{npes}$, and the local computation cost will be $O \left (\frac {m \cdot n \cdot k}{npes} \right)$.

## NNZ-wise parallel algorithm

For the nnz-wise parallel algorithm, just the non-zero elements are sent to the different processors. A pseudo-CSR (Compressed Sparse Row) matrix (Saad, 2003) is created, storing the non-zero value, the number of its row and the number of its column in the same row. Then, the rows of said pseudo-CSR matrix are distributed among the available processes. With this technique, the characteristics of the sparse matrix are exploited, as the density of null elements is higher, so sending only the non-zero elements should be faster. Additionally, the entire fat vector will be also sent to each processor, so that it can multiply its local non-zero elements by their associated elements from the fat vector.

<img width="1180" height="659" alt="image" src="https://github.com/user-attachments/assets/de143411-db8f-4b8f-bdd0-d0644eef0da2" />


The previous image depicts how the nnz-wise parallel algorithm would work with 3 processes and the example matrices. Process 1 gets the first row of the pseudo-CSR matrix, process 2 gets the second one and process 3 the third one, and the three of them get the full fat vector. Then, each process multiplies their local non-zero element by its associated elements from the fat vector according to the coordinates from the pseudo-CSR matrix, obtaining a partial result of the solution fat vector. Finally, all the local results from the different processes are gathered and summed, obtaining thus the final solution fat vector.

In the previous image's depiction, there is only one non-zero element per row, meaning that each process generates the full result for the solution fat vector’s associated row.

With bigger dimensions, this will not be the case, meaning that the processes will generate partial solutions for some elements from the solution fat vector, so all the partial solutions with the same coordinates should be added.

With the real dimensions, to balance the load, each process receives nnz/npes rows from the pseudo-CSR matrix, receiving nnz%npes (the remainder of nnz/npes) processes one additional row.

This means that, locally, each process has a set of rows from the pseudo-CSR matrix, stored as an (nnz/npes)X3 or ((nnz/npes)+1)X3 matrix, and the fat vector, and will generate an ((nnz/npes)*n)X(k+1) or (((nnz/npes)+1)*n)X(k+1) matrix to store the local results. The reason behind the (k+1) is that each processor will generate k values for each non-zero value as results of the multiplications of the non-zero value by its associated elements from the fat vector, and the number of the row of the non-zero element in the sparse matrix will be added as the last element of the row of the partial solution matrix. Thanks to this, the complete solution fat vector will be able to be reconstructed, so that all the partial solutions having the same coordinates can be added.

Regarding the communication cost, it is expected to be on the order of $\frac {nnz \cdot 3}{npes} + n \cdot k$, and the local computation cost will be $O \left (\frac {nnz \cdot k}{npes} \right)$.

Additionally, to have a comparable serial algorithm that takes advantage of the same characteristic of sparse matrices, a serial algorithm following the described steps for the nnz-wise algorithm, but avoiding the splitting of the pseudo-CSR matrix, thus having only one processor performing the computation, was designed.

Moreover, different communication patterns were tested for this algorithm to see their effect. Specifically three tests were performed: replacing MPI_Ibcast by MPI_Bcast, to check the difference between blocking and non-blocking communications (nnz_wise_Bcast); replacing MPI_Allgatherv by MPI_Gatherv, to see if there is a considerable difference between having the final solution available to all the processes or just the root one (nnz_wise_Gatherv); and replacing MPI_Scatterv and MPI_Gatherv by custom functions using MPI_Send and MPI_Recv in loops (nnz_wise_Send_Recv).

## PETSc

To further test the performance of the algorithms, an external mathematical library was used to replicate the operation. In this case, PETSc was the selected library. (UChicago Argonne, LLC and the PETSc Development Team, 2026)

## Weak scalability

Finally, to test the weak scalability of the problem, a programme which sends the entire sparse matrix and the entire fat vector to each process so that all of them must do the entire computation was implemented. The intention of this programme is to show the increasing communication cost when increasing the number of processes, showing that the underperformance of parallel codes does not usually lie on a problem in the code, but in the communication.

## Code execution

This code was tested and executed in Cranfield University’s Crescent2 (CRanfield Educational Scientific Computing ENvironment for Teaching), a supercomputer running on Linux with Intel Xeon E5 2620 CPUs, having 16 CPU cores per node.

The PBS submission scripts (mpi.sub and PETSc.sub) were adapted from Cranfield University's templates for CR2. The configuration is specific to CR2 and may require changes on other systems, particularly regarding:

  - PBS queues and resource requests
  - Environment modules
  - Module search path
  - Intel MPI options
  - PETSc module name and version
  - Network interfaces
  - Executable and input-file paths

To run the PETSc code, first the PETSc module must be loaded (in CR2, `module use /apps/modules/all module load PETSc`). To compile it, make sure that the makefile is in the same directory as `PETSc.c` and execute `make -s PETSc` (-s to avoid the make messages). Then, submit `PETSc.sub` to the scheduler (`qsub PETSc.sub`), and all the strong scalability tests will be performed automatically.

To run all the other codes, the Intel module must be loaded (in CR2, `module use /apps/modules/all module load intel`). To compile them, for the ones that include the `common.h` file, execute `mpicc ./common.c ./file.c -o file`. If `common.h` is not included, just execute `mpicc ./file.c -o file`. Then, make sure that the matrices files are in the same directory as the coding files and submit `mpi.sub` to the scheduler (`qsub mpi.sub`). In this submission file, the number of CPUs used is changed at runtime using a loop, as well as the matrices files used as an input, so only the algorithm must be changed in the mpirun command. Additionally, to do the weak scalability tests, comment the loop for the strong scalability test and uncomment the other loop, changing the algorithm to the one desired to be tested.

CAUTION FOR CR2. When running the algorithms compiled with the Intel module when the PETSc module is loaded, they do not behave as intended. In order to make them work properly, they must be recompiled with the PETSc module loaded using the same command as with the Intel module.

## Additional files

As additional files, the matrices used to test the code are included, as well as a Python script that automatically checks if all the generated solutions from the parallelised implementations match the serial implementation solution.


## Acknowledgements

Some functions included in this project are based on the code submitted to different Stack Overflow posts by the user Jonathan Dursi (https://stackoverflow.com/users/463827/jonathan-dursi).

<hr>

This code is my submission for Cranfield University's High Performance Technical Computing module assignment for the MSc in Computational and Software Techniques in Engineering - Computational Intelligence for Data Analytics.
