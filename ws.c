/**
 * @file ws.c
 * @author trm_na
 * @brief This programme tests the weak scalability of the parallelisation of a matrix and fat 
 * vector multiplication, broadcasting the entire matrix and fat vector to the different processes.
 * @version 1.0
 * @date 2026-02-04
 * 
 * @copyright Copyright (c) 2026
 * 
 */

/* Includes ------------------------------------------------------------------*/
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include "common.h"

/* Functions -----------------------------------------------------------------*/
/**
 * @brief This programme tests the weak scalability of the parallelisation of a matrix and fat 
 * vector multiplication, the entire matrix and fat vector to the different processes.
 * 
 * The matrix and fat vector are read from a file. Then, the full matrix and fat vector are 
 * broadcasted. The result matrix is obtained performing the matrix multiplication. Finally, some 
 * timings are printed as well as the final result.
 * 
 * @param argc Number of arguments passed to the programme in the command line.
 * @param argv Arguments passed to the programme in the command line (in this programme, it could
 * be the name of the file storing the matrix and the fat vector).
 */
int main(int argc, char *argv[])
{
	/* Variable declaration */
	int npes, myrank, m, n, k;
	float **M, **v, **u;
	double t0, t1, t2, t3, t4, read_time, comp_time, tot_time, comm_time, max_read, max_comp, max_tot, max_comm, avg_comp, avg_comm;
	MPI_Status statuses[5];
	MPI_Request requests[5];

	/* MPI initialisation */
	mpi_initialisation(&argc, &argv, &npes, &myrank);

	/* Start the timer */
	t0 = MPI_Wtime();

	/* 
	* Check if a specific file has been asked to be read to perform the computation from its 
	* matrix and fat vector and store them into memory. This is only performed by process 0. 
	*/
	load_matrix(myrank, argc, argv, &m, &n, &k, &M, &v);

	/* End of the reading timer */
	t1 = MPI_Wtime();

	/* Cast the number of rows and columns of the matrix and the number of columns of the fat vector */
	metadata_cast(&m, &n, &k, requests);

	/* Wait to receive the broadcasted parameters before using it */
	MPI_Wait(&requests[0], &statuses[0]);
    MPI_Wait(&requests[1], &statuses[1]);
    MPI_Wait(&requests[2], &statuses[2]);

	/* 
	* Allocate the memory for the solution fat vector, the original fat vector and the elements 
	* from the matrix sent to each process 
	*/
	matrixmalloc(&u, m, k);
	if (myrank != 0){
        matrixmalloc(&M, m, n);
		matrixmalloc(&v, n, k);
	}

	/* Scatter the matrix row-wise and broadcast the fat vector */
	MPI_Ibcast(&M[0][0], m * n, MPI_FLOAT, 0, MPI_COMM_WORLD, &requests[3]);
	MPI_Ibcast(&v[0][0], n * k, MPI_FLOAT, 0, MPI_COMM_WORLD, &requests[4]);

	/* Wait to receive the broadcasted sparse matrix and fat vector before using it */
	MPI_Wait(&requests[3], &statuses[3]);
    MPI_Wait(&requests[4], &statuses[4]);

	/* Start the timer for the computation time */
	t2 = MPI_Wtime();

	/* Perform the local computations */
	local_comp(m, n, k, &M, &v, &u);

	/* End the timer */
	t3 = MPI_Wtime();

	/* Compute different execution times */
	read_time = t1 - t0;
	comp_time = t3 - t2; 
	tot_time = t3 - t0;
	comm_time = tot_time - read_time - comp_time;

	/* Compute and print the execution times and results */
	print_time_and_sol(&read_time, &max_read, &tot_time, &max_tot, &comm_time, &max_comm, &avg_comm, 
					   &comp_time, &max_comp, &avg_comp, NULL, NULL,NULL, NULL, NULL, NULL, NULL, 
					   npes, myrank, m, k, &u);

	/* Free the addresses from the matrices and fat vectors from the memory */
	freematrix(&M);
	freematrix(&v);
	freematrix(&u);

	/* End the MPI session */
	MPI_Finalize();
}
