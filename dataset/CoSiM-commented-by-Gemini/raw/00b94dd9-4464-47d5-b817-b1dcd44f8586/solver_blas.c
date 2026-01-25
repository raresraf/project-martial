/**
 * @file solver_blas.c
 * @brief A matrix solver implementation using BLAS (Basic Linear Algebra Subprograms).
 *
 * This file contains a function `my_solver` that computes a complex matrix expression
 * by leveraging highly optimized BLAS routines for matrix multiplication and other
 * linear algebra operations.
 */
#include "utils.h"
#include "cblas.h"
#include <string.h>


/**
 * @brief Computes the transpose of a square matrix.
 * @param M The input square matrix of size N*N.
 * @param N The dimension of the matrix.
 * @return A new matrix which is the transpose of M. The caller is responsible for freeing this memory.
 */
static double *get_transpose(double *M, int N)
{
	double *tr = calloc(N * N, sizeof(double));
	/**
	 * @brief This nested loop iterates through the matrix to perform the transposition.
	 * Invariant: After each inner loop, one row of the transpose matrix is correctly filled.
	 */
	for (int i = 0; i < N; ++i) {
		for (int j = 0; j < N; ++j) {
			tr[i * N + j] = M[j * N + i];
		}
	}
	return tr;
}


/**
 * @brief Solves a matrix equation of the form: (B*A)*B^T + A*A^T using BLAS routines.
 * @param N The dimension of the square matrices A and B.
 * @param A Input matrix A, assumed to be upper triangular.
 * @param B Input matrix B, assumed to be upper triangular.
 * @return A new matrix containing the result of the computation. The caller is responsible for freeing this memory.
 *
 * @b Algorithm:
 * 1. Transposes matrices A and B to get A^T and B^T.
 * 2. Computes `B * A` using `cblas_dtrmm` (triangular matrix multiplication).
 * 3. Computes `(B * A) * B^T` using `cblas_dgemm` (general matrix multiplication).
 * 4. Computes `A * A^T` using `cblas_dtrmm`.
 * 5. Adds the results of step 3 and 4 to produce the final matrix.
 */
double* my_solver(int N, double *A, double *B) {
	printf("BLAS SOLVER\n");

	double *first_mul = calloc(N * N, sizeof(double));
	double *first_mul_aux = calloc(N * N, sizeof(double));
	double *At = get_transpose(A, N);
	double *Bt = get_transpose(B, N);
	
	memcpy(first_mul, A, N * N * sizeof(double));

	
	// Operation: first_mul = B * A, where B is an upper triangular matrix.
	// `first_mul` is initialized with A and overwritten by the result.
	cblas_dtrmm( CblasRowMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit,
		N, N, 1.0, B, N, first_mul, N);

	
	// Operation: first_mul_aux = 1 * first_mul * Bt + 0 * first_mul_aux
	// This computes (B * A) * B^T.
	 cblas_dgemm( CblasRowMajor, CblasNoTrans, CblasNoTrans,
	 	N, N, N, 1, first_mul, N, Bt, N, 0, first_mul_aux, N);

	
	// Operation: At = A * At, where A is an upper triangular matrix.
	// This overwrites At with the result of A * A^T.
	cblas_dtrmm(CblasRowMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit,
		N, N, 1.0, A, N, At, N);

	/**
	 * @brief This loop adds the two intermediate results.
	 * first_mul_aux = first_mul_aux + At
	 * Resulting in: (B * A) * B^T + A * A^T
	 */
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			first_mul_aux[i * N + j] += At[i * N + j];
		}
	}

	
	// Clean up allocated memory for intermediate matrices.
	free(first_mul);
	free(At);
	free(Bt);
	return first_mul_aux;
}