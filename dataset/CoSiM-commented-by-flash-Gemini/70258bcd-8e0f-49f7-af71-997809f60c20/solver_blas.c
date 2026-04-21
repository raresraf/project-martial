/**
 * @70258bcd-8e0f-49f7-af71-997809f60c20/solver_blas.c
 * @brief Matrix expression solver leveraging optimized Level 3 BLAS.
 * Functional Utility: Computes Result = (A * B) * B^T + (A^T * A) using TRMM for 
 * triangular matrix factors and GEMM for dense product accumulation.
 * Domain: HPC Numerical Optimization.
 */

#include "utils.h"
#include <string.h>
#include "cblas.h"


/**
 * @brief High-performance solver kernel utilizing BLAS routine delegation.
 * Logic: Optimizes the computation of A*B and A^T*A by exploiting the upper 
 * triangular property of matrix A.
 */
double* my_solver(int N, double *A, double *B) {
	printf("BLAS SOLVER\n");

	double *AB = (double*) calloc(N * N, sizeof(double));
	if (AB == NULL) {
		printf("Error calloc");
		return NULL;
	}
	double *ABBt = (double*) calloc(N * N, sizeof(double));
	if (ABBt == NULL) {
		printf("Error calloc");
		return NULL;
	}

	double *AtA = (double*) calloc(N * N, sizeof(double));
	if (AtA == NULL) {
		printf("Error calloc");
		return NULL;
	}

	double *C = (double*) calloc(N * N, sizeof(double));
	if (C == NULL) {
		printf("Error calloc");
		return NULL;
	}

	
	// Pre-condition: Prepare operand buffer for A * B.
	memcpy(AB, B, N * N * sizeof(double));

	/**
	 * Block Logic: Compute AB = A * B.
	 * Optimization: Uses cblas_dtrmm for the left-hand upper triangular matrix A.
	 */
	cblas_dtrmm(CblasRowMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit, N, N, 1.0, A, N, AB, N);

	/**
	 * Block Logic: Compute AtA = A^T * A.
	 * Functional Utility: Dense matrix-matrix multiplication to calculate the Gramian matrix.
	 */
	cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, N, N, N, 1.0, A, N, A, N, 0.0, AtA, N);

	
	// Pre-condition: Aggregate result buffer with the Gramian term.
	memcpy(C, AtA, N * N * sizeof(double));

	/**
	 * Block Logic: Final accumulation: C = AB * B^T + C.
	 * Optimization: Uses DGEMM with beta=1.0 for efficient result merging.
	 */
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, N, N, N, 1.0, AB, N, B, N, 1.0, C, N);

	free(AB);
	free(ABBt);
	free(AtA);

	return C;
}
