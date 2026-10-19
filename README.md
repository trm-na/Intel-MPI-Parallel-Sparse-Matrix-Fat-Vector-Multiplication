# Intel MPI Parallel Sparse-Matrix Fat-Vector Multiplication

This repository includes the code to parallelise a sparse matrix and fat vector multiplication using a distributed memory architecture and Intel MPI as the parallelisation tool.

The fat vector joins several vectors into one matrix to reduce the number of matrix and vector multiplications, computing only a single matrix multiplication to test the data with different parameters.

Regarding the parallelisation methods, 3 were designed, apart from the serial approach. For their description, for illusttration purposes, 3X3 matrix will be used as the sparse matrix and a 3X2 matrix as the fat vector, but these dimensions differ to the ones used for the computations. To refer to the real dimensions in a general way, the sparse matrix is a mXn matrix and the fat vector a nXk matrix. Additionally, the number of processes will be illustrated as 3, whereas this number will vary, and it will be referenced as npes.

### Serial algorithm

First, the problem was approached serially.
The serial solution for this problem is the regular matrix multiplication algorithm: the ith column of the sparse matrix is multiplied by the jth column of the fat vector, and the result of that multiplication will be the element (i, j) of the solution fat vector.

<img width="1024" height="231" alt="image" src="https://github.com/user-attachments/assets/6fa5260b-03a1-4915-b73b-e12ff5a71141" />

The previous image shows how the serial algorithm works to obtain the first element of the solution vector, where $u_{1,1} = M_{1,1} \cdot v_{1,1} + M_{1,2} \cdot v_{2,1} + M_{1,3} \cdot v_{3,1}$. This serial algorithm would go on, first computing all the elements from the first row of the solution fat vector and then continuing in the same way for the subsequent rows. Additionally, as it is a serial algorithm, all the computation would be done by the same CPU core.
Once the serial algorithm was determined, its parallelisation was designed. Three different parallelisation methods were considered for this problem to compare their performance and determine the parallel algorithm that best suits this computation. The programmes are called: row-wise, column-wise and nnz-wise, where nnz stands for number of non-zero elements.

### Row-wise parallel algorithm

As the name states, for the row-wise parallel algorithm, the different rows of the sparse matrix are distributed among the available processes. All the elements of each row are sent to a specific CPU core to compute its k corresponding results, which means that the whole fat vector will be also sent to each processor.

<img width="1181" height="661" alt="image" src="https://github.com/user-attachments/assets/7f74f3a1-c988-418e-a0a2-0552d6c97eb2" />

The previous image depicts how the row-wise parallel algorithm would work with 3 processes and the example matrices. Process 1 gets the first row of the sparse matrix, process 2 gets the second one and process 3 the third one, and the three of them get the full fat vector. Then, each process multiplies their local row by the fat vector, obtaining all the results for that row, and finally, all the local results from the different processes are gathered into the final solution fat vector.
With the real dimensions, to balance the load, each process receives m/npes rows, receiving m%npes (the remainder of m/npes) processes one additional row.
This means that, locally, each process has a set of rows from the sparse matrix, stored as an (m/npes)Xn or ((m/npes)+1)Xn matrix, and the fat vector, and will generate an ((m/npes)*n)Xk or (((m/npes)+1)*n)Xk matrix to store the local results.
Regarding the communication cost, it is expected to be on the order of 𝑚∗𝑛𝑛𝑝𝑒𝑠+𝑛∗𝑘, and the local computation cost will be O(𝑚∗𝑛/𝑛𝑝𝑒𝑠).

This code was tested and executed in Cranfield University’s Crescent2 (CRanfield Educational Scientific Computing ENvironment for Teaching), a supercomputer running on Linux with Intel Xeon E5 2620 CPUs, having 16 CPU cores per node.

The PBS submission scripts were adapted from Cranfield University's templates for CR2. The configuration is specific to CR2 and may require changes on other systems, particularly regarding:

  PBS queues and resource requests
  Environment modules
  Module search path
  Intel MPI options
  PETSc module name and version
  Network interfaces
  Executable and input-file paths

This code is my submission for Cranfield University's High Performance Technical Computing module assignment for the MSc in Computational and Software Techniques in Engineering - Computational Intelligence for Data Analytics.
