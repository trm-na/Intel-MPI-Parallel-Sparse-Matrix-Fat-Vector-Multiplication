/**
 * @file serial.c
 * @author trm_na
 * @brief This programme performs serially a matrix and fat vector multiplication.
 * @version 1.0
 * @date 2026-01-26
 * 
 * @copyright Copyright (c) 2026
 * 
 */

/* Includes ------------------------------------------------------------------*/
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include "common.h"

/* Functions -----------------------------------------------------------------*/
/**
 * @brief This programme performs serially a matrix and fat vector multiplication.
 * 
 * The matrix and fat vector are read from a file. Then, the computations are performed and the 
 * result matrix is obtained. Finally, some timings are printed as well as the final result.
 * 
 * @param argc Number of arguments passed to the programme in the command line.
 * @param argv Arguments passed to the programme in the command line (in this programme, it could
 * be the name of the file storing the matrix and the fat vector).
 */
int main(int argc, char *argv[])
{
	/* Variable declaration */
	int m, n, k;
	float **M, **v, **u;
	clock_t t0, t1, t2, t3;
	double read_time, comp_time, tot_time;
	char filename[256] = "matrices_1.tx";	/* Define the default file to read as matrices_1.tx */
	FILE *fptr;

	/* Start the timer */
	t0 = clock();

	/* 
	* Check if a specific file has been asked to be read to perform the computation from its 
	* matrix and fat vector, store them into memory and create the pseudo CSR matrix.
	*/
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
	read_matrix(fptr, M, m, n);
	read_matrix(fptr, v, n, k);

	/* Close the file */
	fclose(fptr);

	/* End of the reading timer */
	t1 = clock();

	/* Start the timer for the computation time */
	t2 = clock();

	/* Allocate the memory for the solution fat vector */
	matrixmalloc(&u, m, k);

	/* Initialise the final result fat vector as a zero matrix */
	for (int i = 0; i < m; i++){
		for (int j = 0; j < k; j++){
			u[i][j] = 0.0;
		}
	}

	/* Update each fat vector element by adding the partial results obtained */
	for (int i = 0; i < m; i++){
		for (int j = 0; j < k; j++){
			for(int l = 0; l < n; l++){
				u[i][j] += M[i][l] * v[l][j];
			}
		}
	}

	/* End the timer for the local computation time */
	t3 = clock();

	/* Compute different execution times */
	read_time = ((double)t1 - t0)/CLOCKS_PER_SEC;
	comp_time = ((double)t3 - t2)/CLOCKS_PER_SEC; 
	tot_time = ((double)t3 - t0)/CLOCKS_PER_SEC;

	/* Print the execution times and results */
	printf("Total execution time: %f\n", tot_time);
	printf("File reading time: %f\n", read_time);
	printf("Computation time: %f\n", comp_time);
	printf("\n");
	printf("Solution:\n");
	for (int i = 0; i < m; i++) {
		printf("|");
		for (int j = 0; j < k; j++) {
			printf("%6.3f", u[i][j]);
			if (j != k - 1){
				printf("\t");
			}
		}
		printf("|\n");
	}
	

	/* Free from the memory the addresses for the matrices and fat vectors */
	freematrix(&M);
	freematrix(&v);
	freematrix(&u);

	return 0;
}
