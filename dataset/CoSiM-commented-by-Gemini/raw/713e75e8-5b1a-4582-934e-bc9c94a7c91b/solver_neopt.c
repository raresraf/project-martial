
/**
 * @file solver_neopt.c
 * @brief A non-optimized, modular implementation of a matrix solver.
 *
 * This file provides a basic C implementation for a series of matrix operations.
 * Unlike other non-optimized versions, this one is structured with clear helper
 * functions for each distinct matrix operation (transpose, multiply, add),
 * making the logic more readable.
 */
#include "utils.h"


/**
 * @brief Computes the transpose of a square matrix.
 * @param N The dimension of the matrix.
 * @param a The input matrix.
 * @return A new matrix containing the transpose of a.
 */
double* transpose(int N, double *a){
	double* transpose = (double*)malloc((N * N) * sizeof(double));

	for (int i = 0; i < N; i++)
  		for (int j = 0; j < N; j++)
    		transpose[j*N +i] = a[i*N + j];
  		

  	return transpose;
}

/**
 * @brief Performs a naive matrix-matrix multiplication.
 * @param N The dimension of the matrices.
 * @param a The first input matrix.
 * @param b The second input matrix.
 * @return A new matrix containing the result of a * b.
 */
double* multiply(int N, double *a, double* b) {
	double* c = (double*)calloc(sizeof(double), (N * N));

	for (int i = 0 ; i < N ; i++){
		for (int j = 0 ; j < N ; j++){
	    	c[i*N + j] = 0.0;
	    	for (int k = 0 ; k < N ; k++){
				c[i*N + j] += a[i*N + k] * b[k*N + j];
	      	}
   		}
	}

	return c;
}

// Utility function to find the minimum of two integers.
int min(int a, int b){
	return (a < b) ? a : b;
}

/**
 * @brief Performs a non-standard triangular matrix multiplication.
 * @param N The dimension of the matrices.
 * @param a The first input matrix.
 * @param b The second input matrix.
 * @return A new matrix containing the result.
 * @note The loop bound `k < min(i + 1, j + 1)` implements a non-standard
 *       multiplication logic, not a typical triangular matrix multiplication.
 *       The name `triunghiular` is Romanian for "triangular".
 */
double* multiply_triunghiular(int N, double *a, double* b) {
	double* c = (double*)malloc((N * N) * sizeof(double));

	for (int i = 0 ; i < N ; i++){
		for (int j = 0 ; j < N ; j++){
	    	c[i*N + j] = 0.0;
	    	for (int k = 0 ; k < min(i + 1, j + 1) ; k++){
				c[i*N + j] += a[i*N + k] * b[k*N + j];
	      	}
   		}
	}

	return c;
}

/**
 * @brief Performs an in-place addition of two matrices (A = A + B).
 * @param N The dimension of the matrices.
 * @param A The first matrix, which is modified in-place.
 * @param B The second matrix.
 * @return A pointer to the modified matrix A.
 */
double* add(int N, double *A, double *B){
	for(int i = 0 ; i < N ; i++){
		for(int j = 0; j < N ; j++){
			A[i*N + j] += B[i*N + j];
		}
	}
	return A;
}

/**
 * @brief Performs a sequence of matrix operations using modular helper functions.
 *
 * @param N The dimension of the square matrices A and B.
 * @param A A pointer to the first input matrix (N x N).
 * @param B A pointer to the second input matrix (N x N).
 * @return A pointer to the resulting matrix.
 *
 * @note This function computes the expression: C = (A * B) * B^T + A^T * A.
 *       It explicitly breaks down the computation into clear, sequential steps
 *       using dedicated helper functions for each operation.
 */
double* my_solver(int N, double *A, double* B) {
	// Step 1: Explicitly compute the transpose of B and A.
	double* b_transpose = transpose(N, B);
	double* a_transpose = transpose(N, A);

	// Step 2: Compute the intermediate product a_b = A * B.
	double* a_b = multiply(N, A, B);
	// Step 3: Compute a_b_b = (A * B) * B^T.
	double* a_b_b = multiply(N, a_b, b_transpose);

	// Step 4: Compute a_a = A^T * A using a custom triangular multiplication.
	double* a_a = multiply_triunghiular(N, a_transpose, A);

	// Step 5: Perform the final addition.
	double* c = add(N, a_b_b, a_a);

	// Cleanup intermediate allocations.
	free(a_transpose);
	free(b_transpose);
	free(a_b);
	free(a_a);

	return c;
}
