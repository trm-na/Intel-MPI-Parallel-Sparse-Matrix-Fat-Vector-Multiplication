/**
 * @file common.c
 * @author trm_na
 * @brief This programme includes some common functions used by the different matrix-fat vector 
 * multiplication parallelisations.
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
 * @brief Reads a matrix from a file, knowing the number of rows and columns, stores it
 * in memory and returns the number of non-zero elements.
 *
 * @param fptr The pointer of the file at the position of the first element of the matrix.
 * @param matrix The pointer for storing the matrix.
 * @param m The number of rows.
 * @param n The number of columns.
 * @return int The number of non-zero elements.
 */
int read_matrix(FILE *fptr, float **matrix, int m, int n){
	int nnz = 0;
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < n; j++) {
			fscanf(fptr, "%f", &matrix[i][j]);
			if (matrix[i][j] != 0.0){
				++nnz;
			}
		}
	}
	return nnz;
}

/**
 * @brief Allocates the necessary space in memory to store a matrix, having all the elements 
 * stored contiguously.
 * 
 * @param M The pointer for storing the matrix.
 * @param m The number of rows.
 * @param n The number of columns.
 */
void matrixmalloc(float ***M, int m, int n){
	/* 
	* Make a pointer with the size of all the elements contained in the matrix so that they are 
	* contiguously stored 
	*/
	float *Mdata = (float *)malloc(sizeof(float) * m * n);
	(*M) = (float **)malloc(m * sizeof(float *));
	for (int i = 0; i < m; i++) {
		/* 
		* Assign to each matrix row in the original matrix pointer its associated memory address 
		* from the previous pointer 
		*/
		(*M)[i] = &(Mdata[i * n]);
	}
}

/**
 * @brief Frees the memory needed to store a matrix.
 * 
 * @param M The matrix pointer.
 */
void freematrix(float ***M){
	/* Free the memory of the first element of the matrix */
    free(&((*M)[0][0]));

    /* Free the pointers from memory */
    free(*M);
}

/**
 * @brief Creates a pseudo Compressed Sparse Row (CSR) matrix from a sparse matrix.
 * 
 * @param CSR The pointer for storing the CSR matrix.
 * @param M The sparse matrix pointer.
 * @param m The number of rows.
 * @param n The number of columns.
 */
void create_CSR(float ***CSR, float ***M, int m, int n){
	int nnz = 0;
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < n; j++) {
			/* If the element is non-zero, add an element to the CSR matrix */
			if ((*M)[i][j] != 0.0){
				/* The first element is the value */
				(*CSR)[nnz][0] = (*M)[i][j];
				/* The second element is the row */
				(*CSR)[nnz][1] = (float) i;
				/* The third element is the column */
				(*CSR)[nnz][2] = (float) j;
				++nnz;
			}
		}
	}
}

/**
 * @brief Initialises the MPI session.
 * 
 * @param argc Number of arguments passed to the programme in the command line.
 * @param argv Arguments passed to the programme in the command line (in this programme, it could
 * be the name of the file storing the matrix and the fat vector).
 * @param npes  Pointer for the number of processes.
 * @param myrank Pointer for the local process.
 */
void mpi_initialisation(int *argc, char ***argv, int *npes, int *myrank){
    MPI_Init(argc, argv);
	MPI_Comm_size(MPI_COMM_WORLD, npes);
	MPI_Comm_rank(MPI_COMM_WORLD, myrank);
}

/**
 * @brief Loads the matrix from the specified or default file into memory.
 * 
 * @param myrank Number of the local process.
 * @param argc Number of arguments passed to the programme in the command line.
 * @param argv Arguments passed to the programme in the command line (in this programme, it could
 * be the name of the file storing the matrix and the fat vector).
 * @param m The number of rows of the matrix pointer.
 * @param n The number of columns of the matrix and number of rows of the fat vector pointer.
 * @param k The number of columns of the fat vector pointer.
 * @param M The sparse matrix pointer.
 * @param v The fat vector pointer.
 */
void load_matrix(int myrank, int argc, char **argv, int *m, int *n, int *k, float ***M, float ***v){
    if (myrank == 0){
		char filename[256] = "matrices_1.tx";	/* Define the default file to read as matrices_1.tx */
		FILE *fptr;
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
		fscanf(fptr, "%d %d %d", m, n, k);

		/* Allocate the memory for the matrix and fat vector */
		matrixmalloc(M, *m, *n);
    	matrixmalloc(v, *n, *k);

		/* Read the matrix and fat vector and store them into their memory addresses */
		read_matrix(fptr, *M, *m, *n);
		read_matrix(fptr, *v, *n, *k);

		/* Close the file */
		fclose(fptr);
	}
}

/**
 * @brief Broadcasts the number of rows and columns of the matrix and fat vector.
 * 
 * @param m The number of rows of the matrix pointer.
 * @param n The number of columns of the matrix and number of rows of the fat vector pointer.
 * @param k The number of columns of the fat vector pointer.
 * @param requests The requests pointer.
 */
void metadata_cast(int *m, int *n, int *k, MPI_Request *requests){
    MPI_Ibcast(m, 1, MPI_INT, 0, MPI_COMM_WORLD, &requests[0]);
	MPI_Ibcast(n, 1, MPI_INT, 0, MPI_COMM_WORLD, &requests[1]);
	MPI_Ibcast(k, 1, MPI_INT, 0, MPI_COMM_WORLD, &requests[2]);
}

/**
 * @brief Performs the local computations of the scattered elements.
 * 
 * @param m The number of rows of the local matrix.
 * @param n The number of columns of the local matrix and number of rows of the local fat vector.
 * @param k The number of columns of the local fat vector.
 * @param M The sparse matrix pointer.
 * @param v The fat vector pointer.
 * @param sums The partial sums matrix pointer.
 */
void local_comp(int m, int n, int k, float ***M, float ***v, float ***sums){
    float sum;
    for (int i = 0; i < m; i++){
		for (int j = 0; j < k; j++){
			sum = 0.0;
			for(int l = 0; l < n; l++){
				/* 
				* Multiply all the elements from the local matrix by the elements from the 
				* local fat vector 
				*/
				sum += (*M)[i][l] * (*v)[l][j];
			}
			(*sums)[i][j] = sum;
		}
	}
}

/**
 * @brief Obtains the maximum and average values for the execution times and prints them and the 
 * result.
 * 
 * @param read_time Local read time pointer.
 * @param max_read Maximum read time pointer.
 * @param tot_time Local total execution time pointer.
 * @param max_tot Maximum execution time pointer.
 * @param comm_time Local communication time pointer.
 * @param max_comm Maximum communication time pointer.
 * @param avg_comm Average communication time pointer.
 * @param comp_time Local computation time pointer.
 * @param max_comp Maximum computation time pointer.
 * @param avg_comp Average computation time pointer.
 * @param comp_time_no_reduce Local computation time without considering the reduce pointer.
 * @param max_comp_no Maximum computation time without considering the reduce pointer.
 * @param comp_time_reduce Local computation time considering the reduce pointer.
 * @param avg_comp_no Average computation time considering the reduce pointer.
 * @param mat_time Local solution matrix creation time pointer.
 * @param max_mat Maximum solution matrix creation time pointer.
 * @param avg_mat Average solution matrix creation time pointer.
 * @param npes Number of processes.
 * @param myrank Number of the local process.
 * @param m The number of rows of the solution fat vector.
 * @param k The number of columns of the solution fat vector.
 * @param u The solution fat vector pointer.
 */
void print_time_and_sol(double *read_time, double *max_read, double *tot_time, double *max_tot, 
                        double *comm_time, double *max_comm, double *avg_comm, double *comp_time,
                        double *max_comp, double *avg_comp, double *comp_time_no_reduce, 
                        double *max_comp_no, double *comp_time_reduce, double *avg_comp_no,
						double *mat_time, double *max_mat, double *avg_mat, 
						int npes, int myrank, int m, int k, float ***u){
    /* Obtain the maximum values for the execution times and the average values for some of them */
	MPI_Reduce(read_time, max_read, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
	MPI_Reduce(tot_time, max_tot, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
	MPI_Reduce(comm_time, max_comm, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(comm_time, avg_comm, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
	*avg_comm /= npes;
    if (comp_time != NULL && *comp_time > 0){
        MPI_Reduce(comp_time, max_comp, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(comp_time, avg_comp, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        *avg_comp /= npes;
    }
    else{
        MPI_Reduce(comp_time_no_reduce, max_comp_no, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
	    MPI_Reduce(comp_time_reduce, max_comp, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(comp_time_no_reduce, avg_comp_no, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(comp_time_reduce, avg_comp, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        *avg_comp /= npes;
        *avg_comp_no /= npes;
    }
	if (mat_time != NULL && *mat_time > 0){
		MPI_Reduce(mat_time, max_mat, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
		MPI_Reduce(mat_time, avg_mat, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		*avg_mat /= npes;
	}

	/* Process 0 prints the execution times and the solution */
	if (myrank == 0){
		printf("Total execution time: %f\n", *max_tot);
		printf("Maximum file reading time: %f\n", *max_read);
        if (comp_time != NULL && *comp_time > 0){
            printf("Maximum computation time: %f\n", *max_comp);
            printf("Average computation time: %f\n", *avg_comp);
        }
        else{
            printf("Maximum computation time without considering the Allreduce: %f\n", *max_comp_no);
            printf("Maximum computation time considering the Allreduce: %f\n", *max_comp);
            printf("Average computation time without considering the Allreduce: %f\n", *avg_comp_no);
            printf("Average computation time considering the Allreduce: %f\n", *avg_comp);
        }
		if (mat_time != NULL && *mat_time > 0){
            printf("Maximum solution matrix creation time: %f\n", *max_mat);
            printf("Average solution matrix creation time: %f\n", *avg_mat);
        }
		printf("Maximum communication time: %f\n", *max_comm);
		printf("Average communication time: %f\n", *avg_comm);
		printf("\n");
		printf("Solution:\n");
		for (int i = 0; i < m; i++) {
			printf("|");
			for (int j = 0; j < k; j++) {
				printf("%6.3f", (*u)[i][j]);
				if (j != k - 1){
					printf("\t");
				}
			}
			printf("|\n");
		}
	}
}
