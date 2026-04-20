/**
 * @file solver_blas.c
 * @brief BLAS-based optimized matrix solver computing $C = A \times B \times B^T + A^T \times A$.
 * Algorithm: Relies on optimized numerical linear algebra kernels (DGEMM, DTRMM).
 * Time Complexity: $O(N^3)$ due to dense matrix multiplications.
 * Space Complexity: $O(N^2)$ for storing the result matrix C.
 */

#include "utils.h"
#include "cblas.h"

double* my_solver(int N, double *A, double *B) {

	printf("BLAS SOLVER\n");

	double *C = (double *)calloc(N*N,sizeof(double));
	
    /**
     * Block Logic: Evaluates the equation step-by-step using level 3 BLAS operations.
     * 1. cblas_dgemm computes $C = B \times B^T$.
     * 2. cblas_dtrmm computes $C = A \times C$ exploiting A's upper-triangular structure.
     * 3. cblas_dgemm computes $C = A^T \times A + C$.
     */
	cblas_dgemm(CblasRowMajor,CblasNoTrans, CblasTrans, N, N, N, 1, B, N, B, N, 0, C, N);
	cblas_dtrmm(CblasRowMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit, N, N, 1, A, N, C, N);
	cblas_dgemm(CblasRowMajor,CblasTrans, CblasNoTrans, N, N, N, 1, A, N, A, N, 1, C, N);

	return C;
}
