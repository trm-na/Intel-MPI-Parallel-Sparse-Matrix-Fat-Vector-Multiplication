/**
 * @file common.h
 * @author trm_na
 * @brief Header for common.c file.
 * @version 1.0
 * @date 2026-01-18
 * 
 * @copyright Copyright (c) 2026
 * 
 */

 /* Includes ------------------------------------------------------------------*/
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

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
int read_matrix(FILE *fptr, float **matrix, int m, int n);

/**
 * @brief Allocates the necessary space in memory to store a matrix, having all the elements 
 * stored contiguously.
 * 
 * @param M The pointer for storing the matrix.
 * @param m The number of rows.
 * @param n The number of columns.
 */
void matrixmalloc(float ***M, int m, int n);

/**
 * @brief Frees the memory needed to store a matrix.
 * 
 * @param M The matrix pointer.
 */
void freematrix(float ***M);

/**
 * @brief Creates a pseudo Compressed Sparse Row (CSR) matrix from a sparse matrix.
 * 
 * @param CSR The pointer for storing the CSR matrix.
 * @param M The sparse matrix pointer.
 * @param m The number of rows.
 * @param n The number of columns.
 */
void create_CSR(float ***CSR, float ***M, int m, int n);

/**
 * @brief Initialises the MPI session.
 * 
 * @param argc Number of arguments passed to the programme in the command line.
 * @param argv Pointer for the arguments passed to the programme in the command line (in this 
 * programme, it could be the name of the file storing the matrix and the fat vector).
 * @param npes  Pointer for the number of processes.
 * @param myrank Pointer for the local process.
 */
void mpi_initialisation(int *argc, char ***argv, int *npes, int *myrank);

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
void load_matrix(int myrank, int argc, char **argv, int *m, int *n, int *k, float ***M, float ***v);

/**
 * @brief Broadcasts the number of rows and columns of the matrix and fat vector.
 * 
 * @param m The number of rows of the matrix pointer.
 * @param n The number of columns of the matrix and number of rows of the fat vector pointer.
 * @param k The number of columns of the fat vector pointer.
 * @param requests The requests pointer.
 */
void metadata_cast(int *m, int *n, int *k, MPI_Request *requests);

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
void local_comp(int m, int n, int k, float ***M, float ***v, float ***sums);

/**
 * @brief Obtains the maximum and average values for the execution times and prints them and the result.
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
                        int npes, int myrank, int m, int k, float ***u);