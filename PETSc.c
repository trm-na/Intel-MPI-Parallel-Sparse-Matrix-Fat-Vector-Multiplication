/**
 * @file PETSc.c
 * @author trm_na
 * @brief This programme parallelises matrix and fat vector multiplication using the PETSc library.
 * @version 1.0
 * @date 2026-02-01
 * 
 * @copyright Copyright (c) 2026
 * 
 */

/* Includes ------------------------------------------------------------------*/
#include <mpi.h>
#include <petscsys.h>
#include <petscmat.h>
#include <stdio.h>
#include <stdlib.h>

/* Functions -----------------------------------------------------------------*/
/**
 * @brief This programme parallelises matrix and fat vector multiplication using the PETSc library.
 * 
 * The matrix and fat vector are read from a file. Then, the PETSc matrices are created and the 
 * multiplication is performed. Finally, some timings are printed as well as the final result.
 * 
 * @param argc Number of arguments passed to the programme in the command line.
 * @param argv Arguments passed to the programme in the command line (in this programme, it could
 * be the name of the file storing the matrix and the fat vector).
 */
int main(int argc, char *argv[])
{
	/* Variable declaration */
  	PetscMPIInt myrank, npes;
	PetscInt m = 0, n = 0, k = 0;
	Mat M, v, u, u_seq;
	PetscScalar petsc_val, *data;
	float val;
	double t0, t1, t2, t3, read_time, comp_time, tot_time, comm_time, max_read, max_comp, max_tot, max_comm, avg_comp, avg_comm;
	char filename[256] = "matrices_1.tx";	/* Define the default file to read as matrices_1.tx */
	FILE *fptr;

	/* PETSc initialisation */
	PetscCall(PetscInitialize(&argc, &argv, NULL, NULL));
  	PetscCall(MPI_Comm_rank(PETSC_COMM_WORLD, &myrank));
	PetscCall(MPI_Comm_size(PETSC_COMM_WORLD, &npes));

	/* Start the timer */
	t0 = MPI_Wtime();


	/* 
	* Check if a specific file has been asked to be read to perform the computation from its 
	* matrix and fat vector and store their dimensions into memory. This is only performed by 
	* process 0. 
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
	}

	/* Cast the number of rows and columns of the matrix and the number of columns of the fat vector */
	PetscCall(MPI_Bcast(&m, 1, MPI_INT, 0, PETSC_COMM_WORLD));
	PetscCall(MPI_Bcast(&n, 1, MPI_INT, 0, PETSC_COMM_WORLD));
	PetscCall(MPI_Bcast(&k, 1, MPI_INT, 0, PETSC_COMM_WORLD));

	/* Initialise the matrices */
	PetscCall(MatCreate(PETSC_COMM_WORLD, &M));
	PetscCall(MatSetSizes(M, PETSC_DECIDE, PETSC_DECIDE, m, n));	/* PETSc decides the number of local rows and columns */
	PetscCall(MatSetFromOptions(M));	/* Default type: CSR*/
	PetscCall(MatSetUp(M));				/* Allocate internal structures */
	
	PetscCall(MatCreateDense(PETSC_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE, n, k, NULL, &v));	/* PETSc decides the number of local rows and columns */
	PetscCall(MatSetUp(v));

	/* 
	* Read the values of the matrix and fat vector and store them into memory. 
	* This is only performed by process 0. 
	*/
	if (myrank == 0){
		/* Read the matrix and store it into memory */
		for (int i = 0; i < m; i++) {
			for (int j = 0; j < n; j++) {
				fscanf(fptr, "%f", &val);
				if (val != 0.0){
					petsc_val = (PetscScalar)val;
					PetscCall(MatSetValue(M, i, j, petsc_val, INSERT_VALUES));
				}
			}
		}

		/* Read the fat vector and store it into memory */
		for (int i = 0; i < n; i++) {
			for (int j = 0; j < k; j++) {
				fscanf(fptr, "%f", &val);
				if (val != 0.0){
					petsc_val = (PetscScalar)val;
					PetscCall(MatSetValue(v, i, j, petsc_val, INSERT_VALUES));
				}
			}
		}

		/* Close the file */
		fclose(fptr);
	}

	/* End of the reading timer */
	t1 = MPI_Wtime();

	/* Assemble M and v so that they can be used */
	PetscCall(MatAssemblyBegin(M, MAT_FINAL_ASSEMBLY));
	PetscCall(MatAssemblyEnd(M, MAT_FINAL_ASSEMBLY));
	PetscCall(MatAssemblyBegin(v, MAT_FINAL_ASSEMBLY));
	PetscCall(MatAssemblyEnd(v, MAT_FINAL_ASSEMBLY));

	/* Start the timer for the computation time */
	t2 = MPI_Wtime();

	/* 
	* Perform the multiplication. Create a new matrix for the solution and let PETSc determine 
	* the expected fill 
	*/
	PetscCall(MatMatMult(M, v, MAT_INITIAL_MATRIX, PETSC_DETERMINE, &u));
	
	/* End the timer for the computation time */
	t3 = MPI_Wtime();

	/* Compute different execution times */
	read_time = t1 - t0;
	comp_time = t3 - t2; 
	tot_time = t3 - t0;
	comm_time = tot_time - read_time - comp_time;

	/* Obtain the maximum values for the execution times and the average values for some of them */
	MPI_Reduce(&read_time, &max_read, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
	MPI_Reduce(&tot_time, &max_tot, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
	MPI_Reduce(&comm_time, &max_comm, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&comm_time, &avg_comm, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
	MPI_Reduce(&comp_time, &max_comp, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&comp_time, &avg_comp, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    avg_comp /= npes;
	avg_comm /= npes;

	/* Copy the solution matrix to rank 0 */
	MatCreateRedundantMatrix(u, 0, PETSC_COMM_SELF, MAT_INITIAL_MATRIX, &u_seq);

	/* Process 0 prints the execution times and the solution matrix */
	if (myrank == 0){
		printf("Total execution time: %f\n", max_tot);
		printf("Maximum file reading time: %f\n", max_read);
		printf("Maximum computation time: %f\n", max_comp);
		printf("Average computation time: %f\n", avg_comp);
		printf("Maximum communication time: %f\n", max_comm);
		printf("Average communication time: %f\n", avg_comm);
		printf("\n");
		printf("Solution:\n");
		MatDenseGetArray(u_seq, &data);
		for (PetscInt i = 0; i < m; i++) {
			printf("|");
			for (PetscInt j = 0; j < k; j++) {
				printf("%7.3f", PetscRealPart(data[i + j * m]));
				if (j != k - 1){
					printf("  ");
				}
			}
			printf("|\n");
		}
	}

	/* Free from the memory the addresses for the types, matrices and fat vectors */
	MatDenseRestoreArray(u_seq, &data);
	MatDestroy(&M); 
	MatDestroy(&v); 
	MatDestroy(&u);
	MatDestroy(&u_seq);

	/* End the PETSc session */
	PetscCall(PetscFinalize());
}
