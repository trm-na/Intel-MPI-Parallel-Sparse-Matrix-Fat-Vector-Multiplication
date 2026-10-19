/**
 * @file column_wise.c
 * @author trm_na
 * @brief This programme parallelises matrix and fat vector multiplication distributing 
 * the columns of the matrix to the different processes.
 * 
 * Credits: some functions in this file are based on the code submitted to different Stack Overflow 
 * posts by the user Jonathan Dursi (https://stackoverflow.com/users/463827/jonathan-dursi)
 * 
 * @version 1.0
 * @date 2026-01-06
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
 * the columns of the matrix to the different processes. Specifically, the columns are 
 * scattered among the different available processes.
 * 
 * The matrix and fat vector are read from a file. Then, the columns from the matrix and the 
 * needed elements from the fat vector for each process are scattered. The needed computations
 * are performed and the result matrix is computed doing a reduce operation. Finally, some timings
 * are printed as well as the final result.
 * 
 * @param argc Number of arguments passed to the programme in the command line.
 * @param argv Arguments passed to the programme in the command line (in this programme, it could
 * be the name of the file storing the matrix and the fat vector).
 */
int main(int argc, char *argv[])
{
	/* Variable declaration */
	int npes, myrank, m, n, k, nfull, rem;
	int *Mdispls, *Mscounts, *vdispls, *vscounts;
	float **M, **v, **u, **Mlocal, **vlocal, **sums;
	double t0, t1, t2, t3, t4, read_time, comp_time_no_reduce, comp_time_reduce, tot_time, comm_time, max_read, max_comp_no, max_comp, max_tot, max_comm, avg_comp_no, avg_comp, avg_comm;
	MPI_Datatype Mtype, Mcol, localtype, localcol;
	MPI_Status statuses[3];
	MPI_Request requests[3];

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
	vdispls = (int *)malloc(npes * sizeof(int));
	vscounts = (int *)malloc(npes * sizeof(int));

	/* Wait to receive the broadcasted n before using it */
	MPI_Wait(&requests[1], &statuses[1]);

	/* 
	* Compute the integer number of elements to scatter evenly to the processes and the remaining
	* ones to distribute additionally to some processes
	*/
	nfull = n / npes;
	rem = n % npes;

	/* Wait to receive the broadcasted k before using it */
	MPI_Wait(&requests[2], &statuses[2]);

	/* Create the sendcounts and displacement arrays for the matrix and fat vector */
	for (int i = 0; i < npes; i++){
		if (i < rem){
			/* Add an additional element to send to process i to balance the load */
			Mscounts[i] = nfull + 1;
			vscounts[i] = (nfull + 1) * k;
		}
		else{
			/* 
			* For the remaining processes, send the previously computed number of elements to 
			* have an even distribution 
			*/
			Mscounts[i] = nfull;
			vscounts[i] = nfull * k;
		}
		if (i > 0){
			/* 
			* Displacements for process i are the starting element from previous process plus 
			* the amount of elements sent to it
			*/
			Mdispls[i] = Mdispls[i - 1] + Mscounts[i - 1];
			vdispls[i] = vdispls[i - 1] + vscounts[i - 1];
		}
		else{
			/* Start the displacements at the first element */
			Mdispls[i] = 0;
			vdispls[i] = 0;
		}
	}

	/* Wait to receive the broadcasted m before using it */
	MPI_Wait(&requests[0], &statuses[0]);

	/* 
	* Allocate the memory for the solution fat vector and the elements from the matrix and 
	* fat vector sent to each process 
	*/
	matrixmalloc(&u, m, k);
	matrixmalloc(&Mlocal, m, Mscounts[myrank]);
	matrixmalloc(&vlocal, vscounts[myrank]/k, k);

	/* Create a type to send the columns */
	MPI_Type_vector(m, 1, n, MPI_FLOAT, &Mtype);
    MPI_Type_create_resized(Mtype, 0, sizeof(float), &Mcol);
    MPI_Type_commit(&Mcol);

	/* Create a type to recieve the columns */
	MPI_Type_vector(m, 1, Mscounts[myrank], MPI_FLOAT, &localtype);
    MPI_Type_create_resized(localtype, 0, sizeof(float), &localcol);
	MPI_Type_commit(&localcol);

	/* Scatter the matrix column-wise and the elements needed from the fat vector */
	MPI_Scatterv((myrank==0) ? &M[0][0] : NULL, Mscounts, Mdispls, Mcol, &Mlocal[0][0], Mscounts[myrank], localcol, 0, MPI_COMM_WORLD);
	MPI_Scatterv((myrank==0) ? &v[0][0] : NULL, vscounts, vdispls, MPI_FLOAT, &vlocal[0][0], vscounts[myrank], MPI_FLOAT, 0, MPI_COMM_WORLD);

	/* Allocate the memory for the partial sums of the computations */
	matrixmalloc(&sums, m, k);

	/* Start the timer for the computation time */
	t2 = MPI_Wtime();

	/* Perform the local computations of the scattered elements */
	local_comp(m, Mscounts[myrank], k, &Mlocal, &vlocal, &sums);

	/* End the timer for the local computation time */
	t3 = MPI_Wtime();
	
	/* 
	* Obtain the result by doing a reduce operation. All the elements corresponding to the same 
	* position of the solution fat vector are summed, obtaining thus the final solution 
	*/
	MPI_Allreduce(&sums[0][0], &u[0][0], m*k, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);

	/* End the timer */
	t4 = MPI_Wtime();

	/* Compute different execution times */
	read_time = t1 - t0;
	comp_time_no_reduce = t3 - t2; 
	comp_time_reduce = t4 - t2;
	tot_time = t4 - t0;
	comm_time = tot_time - read_time - comp_time_no_reduce;

	/* Compute and print the execution times and results */
	print_time_and_sol(&read_time, &max_read, &tot_time, &max_tot, &comm_time, &max_comm, &avg_comm,
					   NULL, &max_comp, &avg_comp, &comp_time_no_reduce, &max_comp_no,
					   &comp_time_reduce, &avg_comp_no, NULL, NULL, NULL, npes, myrank, m, k, &u);

	/* Free from the memory the addresses for the types, matrices and fat vectors */
	MPI_Type_free(&Mtype);
	MPI_Type_free(&Mcol);
	MPI_Type_free(&localtype);
	MPI_Type_free(&localcol);
	if(myrank == 0){
		freematrix(&M);
		freematrix(&v);
	}
	freematrix(&Mlocal);
	freematrix(&vlocal);
	freematrix(&u);
	freematrix(&sums);

	/* End the MPI session */
	MPI_Finalize();

	return 0;
}
