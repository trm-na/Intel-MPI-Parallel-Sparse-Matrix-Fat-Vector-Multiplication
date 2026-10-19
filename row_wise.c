/**
 * @file row_wise.c
 * @author trm_na
 * @brief This programme parallelises matrix and fat vector multiplication distributing 
 * the rows of the matrix to the different processes.
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
 * the rows of the matrix to the different processes. Specifically, the rows are 
 * scattered among the different available processes.
 * 
 * The matrix and fat vector are read from a file. Then, the rows from the matrix are scattered and
 * the full fat vector is broadcasted. The needed computations are performed and the result matrix 
 * is obtained gathering all the elements. Finally, some timings are printed as well as the final 
 * result.
 * 
 * @param argc Number of arguments passed to the programme in the command line.
 * @param argv Arguments passed to the programme in the command line (in this programme, it could
 * be the name of the file storing the matrix and the fat vector).
 */
int main(int argc, char *argv[])
{
	/* Variable declaration */
	int npes, myrank, m, n, k, nfull, rem;
	int *Mdispls, *Mscounts, *udispls, *uscounts;
	float **M, **v, **u, **Mlocal, **sums;
	double t0, t1, t2, t3, t4, read_time, comp_time, tot_time, comm_time, max_read, max_comp, max_tot, max_comm, avg_comp, avg_comm;
	MPI_Status statuses[4];
	MPI_Request requests[4];

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

	/* Allocate memory for the displacements and sendcounts arrays for the matrix and fat vector */
	Mdispls = (int *)malloc(npes * sizeof(int));
	Mscounts = (int *)malloc(npes * sizeof(int));
	udispls = (int *)malloc(npes * sizeof(int));
	uscounts = (int *)malloc(npes * sizeof(int));

	/* Wait to receive the broadcasted m before using it */
	MPI_Wait(&requests[0], &statuses[0]);

	/* 
	* Compute the integer number of rows to scatter evenly to the processes and the remaining
	* ones to distribute additionally to some processes
	*/
	nfull = m / npes;
	rem = m % npes;

	/* Wait to receive the broadcasted n and k before using it */
	MPI_Wait(&requests[1], &statuses[1]);
	MPI_Wait(&requests[2], &statuses[2]);

	/* Create the sendcounts and displacement arrays for the matrix and solution fat vector */
	for (int i = 0; i < npes; i++){
		if (i < rem){
			/* Add an additional row to send to/receive from process i to balance the load */
			Mscounts[i] = (nfull + 1) * n;
			uscounts[i] = (nfull + 1) * k;
		}
		else{
			/* 
			* For the remaining processes, send/receive the previously computed number of rows to 
			* have an even distribution 
			*/
			Mscounts[i] = nfull * n;
			uscounts[i] = nfull * k;
		}
		if (i > 0){
			/* 
			* Displacements for process i are the starting element from previous process plus 
			* the amount of elements sent to it
			*/
			Mdispls[i] = Mdispls[i - 1] + Mscounts[i - 1];
			udispls[i] = udispls[i - 1] + uscounts[i - 1];
		}
		else{
			/* Start the displacements at the first element */
			Mdispls[i] = 0;
			udispls[i] = 0;
		}
	}

	/* 
	* Allocate the memory for the solution fat vector, the original fat vector and the elements 
	* from the matrix sent to each process 
	*/
	matrixmalloc(&u, m, k);
	matrixmalloc(&Mlocal, Mscounts[myrank]/n, n);
	if (myrank != 0){
		matrixmalloc(&v, n, k);
	}

	/* Scatter the matrix row-wise and broadcast the fat vector */
	MPI_Scatterv((myrank==0) ? &M[0][0] : NULL, Mscounts, Mdispls, MPI_FLOAT, &Mlocal[0][0], Mscounts[myrank], MPI_FLOAT, 0, MPI_COMM_WORLD);
	MPI_Ibcast(&v[0][0], n * k, MPI_FLOAT, 0, MPI_COMM_WORLD, &requests[3]);

	/* Allocate the memory for the partial sums of the computations */
	matrixmalloc(&sums, Mscounts[myrank], k);

	/* Wait to receive the broadcasted fat vector before using it */
	MPI_Wait(&requests[3], &statuses[3]);

	/* Start the timer for the computation time */
	t2 = MPI_Wtime();

	/* Perform the local computations of the scattered elements */
	local_comp(Mscounts[myrank]/n, n, k, &Mlocal, &v, &sums);

	/* End the timer for the local computation time */
	t3 = MPI_Wtime();
	
	/* 
	* Obtain the result by doing a gather operation. All the elements from the solution fat vector 
	* are gathered from the processes, obtaining thus the final solution 
	*/
	MPI_Allgatherv(&sums[0][0], uscounts[myrank], MPI_FLOAT,  &u[0][0], uscounts, udispls, MPI_FLOAT, MPI_COMM_WORLD);

	/* End the timer */
	t4 = MPI_Wtime();

	/* Compute different execution times */
	read_time = t1 - t0;
	comp_time = t3 - t2; 
	tot_time = t4 - t0;
	comm_time = tot_time - read_time - comp_time;

	/* Compute and print the execution times and results */
	print_time_and_sol(&read_time, &max_read, &tot_time, &max_tot, &comm_time, &max_comm, &avg_comm, 
					   &comp_time, &max_comp, &avg_comp, NULL, NULL,NULL, NULL, NULL, NULL, NULL, npes, 
					   myrank, m, k, &u);

	/* Free the addresses from the matrices and fat vectors from the memory */
	if(myrank == 0){
		freematrix(&M);
	}
	freematrix(&Mlocal);
	freematrix(&v);
	freematrix(&u);
	freematrix(&sums);

	/* End the MPI session */
	MPI_Finalize();
}
