/**
 * @file nnz_wise.c
 * @author trm_na
 * @brief This programme parallelises matrix and fat vector multiplication distributing 
 * the non-zero elements of the matrix to the different processes.
 * 
 * Credits: some functions in this file are based on the code submitted to different Stack Overflow 
 * posts by the user Jonathan Dursi (https://stackoverflow.com/users/463827/jonathan-dursi)
 * 
 * @version 1.0
 * @date 2026-01-08
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
 * @brief This programme parallelises matrix and fat vector multiplication distributing 
 * the non-zero of the matrix to the different processes. Specifically, the non-zero elements are 
 * scattered among the different available processes.
 * 
 * The matrix and fat vector are read from a file and a pseudo CSR matrix is created. Then, the 
 * CSR matrix is scattered by rows and the full fat vector is broadcasted. The needed computations 
 * are performed and the result matrix is obtained gathering all the elements. Finally, some 
 * timings are printed as well as the final result.
 * 
 * @param argc Number of arguments passed to the programme in the command line.
 * @param argv Arguments passed to the programme in the command line (in this programme, it could
 * be the name of the file storing the matrix and the fat vector).
 */
int main(int argc, char *argv[])
{
	/* Variable declaration */
	int npes, myrank, m, n, k, nfull, rem, nnz;
	int *CSRdispls, *CSRscounts, *soldispls, *solscounts;
	float **M, **v, **u, **res, **CSR, **CSRlocal, **sol;
	double t0, t1, t2, t3, t4, t5, read_time, comp_time, tot_time, comm_time, mat_time, max_read, max_comp, max_tot, max_comm, avg_comp, avg_comm, max_mat, avg_mat;
	char filename[256] = "matrices_1.tx";	/* Define the default file to read as matrices_1.tx */
	FILE *fptr;
	MPI_Status statuses[5];
	MPI_Request requests[5];

	/* MPI initialisation */
	mpi_initialisation(&argc, &argv, &npes, &myrank);

	/* Start the timer */
	t0 = MPI_Wtime();

	/* 
	* Check if a specific file has been asked to be read to perform the computation from its 
	* matrix and fat vector, store them into memory and create the pseudo CSR matrix. This is 
	* only performed by process 0. 
	*/
	if (myrank == 0){
		if (argc > 1) {
			/* If a specific filename has been included in the command line, replace the filename */
			snprintf(filename, sizeof(filename), "%s", argv[1]);
		}

		/* Open the file storing the matrix and fat vector */
		fptr = fopen(filename, "r");

		/* 
		* Collect the number of rows and columns of the matrix and the number of columns 
		* of the fat vector 
		*/
		fscanf(fptr, "%d %d %d", &m, &n, &k);

		/* Allocate the memory for the matrix and fat vector */
		matrixmalloc(&M, m, n);
    	matrixmalloc(&v, n, k);

		/* 
		* Read the matrix and fat vector, store them into their memory addresses and get the number 
		* of non-zero elements 
		*/
		nnz = read_matrix(fptr, M, m, n);
		read_matrix(fptr, v, n, k);

		/* Close the file */
		fclose(fptr);

		/* Allocate the memory for the CSR matrix */
		matrixmalloc(&CSR, nnz, 3);

		/* Create the CSR matrix */
		create_CSR(&CSR, &M, m, n);
	}

	/* End of the reading timer */
	t1 = MPI_Wtime();

	/* 
	* Cast the number of rows and columns of the matrix, the number of columns of the fat vector 
	* and the number of non-zero elements 
	*/
	metadata_cast(&m, &n, &k, requests);
	MPI_Ibcast(&nnz, 1, MPI_INT, 0, MPI_COMM_WORLD, &requests[3]);

	/* Allocate memory for the displacements and sendcounts arrays for the CSR and solution matrices */
	CSRdispls = (int *)malloc(npes * sizeof(int));
	CSRscounts = (int *)malloc(npes * sizeof(int));
	soldispls = (int *)malloc(npes * sizeof(int));
	solscounts = (int *)malloc(npes * sizeof(int));

	/* Wait to receive the broadcasted number of non-zero elements before using it */
	MPI_Wait(&requests[3], &statuses[3]);

	/* 
	* Compute the integer number of non-zero elements to scatter evenly to the processes and the 
	* remaining ones to distribute additionally to some processes
	*/
	nfull = nnz / npes;
	rem = nnz % npes;

	/* Wait to receive the broadcasted k before using it */
	MPI_Wait(&requests[2], &statuses[2]);

	/* Create the sendcounts and displacement arrays for the CSR and solution matrices */
	for (int i = 0; i < npes; i++){
		if (i < rem){
			/* Add an additional element to send to/receive from process i to balance the load */
			CSRscounts[i] = (nfull + 1) * 3;
			solscounts[i] = (nfull + 1) * (k + 1);	/* Each solution matrix row is composed by k values and the row number */
		}
		else{
			/* 
			* For the remaining processes, send/receive the previously computed number of elements 
			* to have an even distribution 
			*/
			CSRscounts[i] = nfull * 3;
			solscounts[i] = nfull * (k + 1);
		}
		if (i > 0){
			/* 
			* Displacements for process i are the starting element from previous process plus 
			* the amount of elements sent to it
			*/
			CSRdispls[i] = CSRdispls[i - 1] + CSRscounts[i - 1];
			soldispls[i] = soldispls[i - 1] + solscounts[i - 1];
		}
		else{
			/* Start the displacements at the first element */
			CSRdispls[i] = 0;
			soldispls[i] = 0;
		}
	}

	/* Wait to receive the broadcasted m before using it */
	MPI_Wait(&requests[0], &statuses[0]);

	/* 
	* Allocate the memory for the solution fat vector, the elements from the CSR matrix sent 
	* to each process and some intermediate solution fat vectors
	*/
	matrixmalloc(&u, m, k);
	matrixmalloc(&CSRlocal, CSRscounts[myrank]/3, 3);
	matrixmalloc(&res, CSRscounts[myrank]/3, (k + 1));
	matrixmalloc(&sol, nnz, (k + 1));

	/* Wait to receive the broadcasted n before using it */
	MPI_Wait(&requests[1], &statuses[1]);
	
	if (myrank != 0){
		matrixmalloc(&v, n, k);
	}

	/* Scatter the CSR matrix row-wise and broadcast the fat vector */
	MPI_Scatterv((myrank==0) ? &CSR[0][0] : NULL, CSRscounts, CSRdispls, MPI_FLOAT, &CSRlocal[0][0], CSRscounts[myrank], MPI_FLOAT, 0, MPI_COMM_WORLD);
	MPI_Ibcast(&v[0][0], n * k, MPI_FLOAT, 0, MPI_COMM_WORLD, &requests[4]);

	/* Wait to receive the broadcasted fat vector before using it */
	MPI_Wait(&requests[4], &statuses[4]);

	/* Start the timer for the computation time */
	t2 = MPI_Wtime();

	/* Perform the local computations of the scattered elements */
	for (int i = 0; i < CSRscounts[myrank]/3; i++){
		for (int j = 0; j < k; j++){
			/* 
			* Multiply all the elements from the local CSR matrix by the corresponding elements 
			* from the fat vector 
			*/
			res[i][j] = CSRlocal[i][0] * v[(int) CSRlocal[i][2]][j];;
		}
		/* Add the row number to the result matrix row at the last place */
		res[i][k] = CSRlocal[i][1];
	}

	/* End the timer for the local computation time */
	t3 = MPI_Wtime();
	
	/* 
	* Obtain the result by doing a gather operation. All the elements from the solution fat vector 
	* are gathered from the processes, obtaining thus a partial solution 
	*/
	MPI_Allgatherv(&res[0][0], solscounts[myrank], MPI_FLOAT,  &sol[0][0], solscounts, soldispls, MPI_FLOAT, MPI_COMM_WORLD);

	/* Start the timer for the final result computation */
	t4 = MPI_Wtime();

	/* Initialise the final result fat vector as a zero matrix */
	for (int i = 0; i < m; i++){
		for (int j = 0; j < k; j++){
			u[i][j] = 0.0;
		}
	}

	/* Update each fat vector element by adding the partial results obtained */
	for (int i = 0; i < nnz; i++){
		for (int j = 0; j < k; j++){
			u[(int) sol[i][k]][j] += sol[i][j];
		}
	}	

	/* End the timer for the final result computation */
	t5 = MPI_Wtime();

	/* Compute different execution times */
	read_time = t1 - t0;
	comp_time = t3 - t2; 
	tot_time = t5 - t0;
	mat_time = t5 - t4;
	comm_time = tot_time - read_time - comp_time - mat_time;

	/* Compute and print the execution times and results */
	print_time_and_sol(&read_time, &max_read, &tot_time, &max_tot, &comm_time, &max_comm, &avg_comm, 
					   &comp_time, &max_comp, &avg_comp, NULL, NULL,NULL, NULL, &mat_time, &max_mat, &avg_mat, 
					   npes, myrank, m, k, &u);

	/* Free from the memory the addresses for the matrices and fat vectors */
	if(myrank == 0){
		freematrix(&M);
	}
	freematrix(&CSRlocal);
	freematrix(&v);
	freematrix(&u);
	freematrix(&res);

	/* End the MPI session */
	MPI_Finalize();
}
