/**
 * @623b273a-0241-4abf-bd8b-e5af273c535b/solver_blas.c
 * @brief High-performance matrix expression solver using Level 3 BLAS.
 * Functional Utility: Solves Result = (A * B) * B^T + (A^T * A) by decomposing the 
 * expression into optimized triangular (TRMM) and general (GEMM) matrix operations.
 * Domain: HPC Numerical Optimization.
 */

#include "utils.h"
#include "cblas.h"


/**
 * @brief Matrix solver kernel leveraging optimized linear algebra subroutines.
 * Logic: Efficiently handles the upper triangular property of matrix A to reduce 
 * redundant floating-point operations.
 */
double* my_solver(int N, double *A, double *B) {
	printf("BLAS SOLVER\n");
	double *AAt;
	double *BBt;
	register int i, j;
	double alpha = 1.0;

	// Functional Utility: Configures BLAS parameters for Row-Major memory layout.
	enum CBLAS_ORDER layout;
	enum CBLAS_TRANSPOSE transa;
	enum CBLAS_TRANSPOSE transat;
	enum CBLAS_SIDE side;
	enum CBLAS_UPLO upper;

	upper = CblasUpper;
	side = CblasLeft;
	layout = CblasRowMajor;
	transa = CblasNoTrans;
	transat = CblasTrans;

	
	AAt = calloc(N * N, sizeof(*AAt));
	BBt = calloc(N * N, sizeof(*BBt));

	/**
	 * Block Logic: Buffer initialization for intermediate results.
	 * Optimization: Uses pointer-based copies for matrix A and B into workspace buffers.
	 */
	for(i = 0; i < N; i++) {
		double *aa_ptr = AAt + i * N;
		double *a_ptr = A + i * N;
		for(j = 0; j < N; j++) {
			*aa_ptr = *a_ptr;
			aa_ptr++;
			a_ptr++; 
		}
	}

	for(i = 0; i < N; i++) {
		double *bb_ptr = BBt + i * N;
		double *b_ptr = B + i * N;
		for(j = 0; j < N; j++) {
			*bb_ptr = *b_ptr;
			bb_ptr++;
			b_ptr++; 
		}
	}
	
	/**
	 * Block Logic: Compute AAt = A^T * A.
	 * Optimization: In-place triangular matrix-matrix multiplication with transposition.
	 */
	cblas_dtrmm(layout, side, upper, transat, CblasNonUnit, N, N, alpha, A, N, AAt, N);

	/**
	 * Block Logic: Compute BBt = A * B.
	 * Optimization: Exploits upper triangularity of A for the left-hand operand.
	 */
	cblas_dtrmm(layout, side, upper, transa, CblasNonUnit, N, N, alpha, A, N, BBt, N);
	
	/**
	 * Block Logic: Final aggregation: AAt = BBt * B^T + AAt.
	 * Functional Utility: General matrix-matrix multiplication adding result to previous AAt.
	 */
	cblas_dgemm(layout, transa, transat, N, N, N, alpha, BBt, N, B, N, alpha, AAt, N);
	
	free(BBt);

	return AAt;
}
