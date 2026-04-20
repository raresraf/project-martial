/**
 * @file solver_blas.c
 * @brief BLAS-based optimized matrix solver computing $C = A \times B \times B^T + A^T \times A$.
 * Algorithm: Relies on optimized numerical linear algebra kernels (DGEMM, DTRMM).
 * Time Complexity: $O(N^3)$ due to dense matrix multiplications.
 * Space Complexity: $O(N^2)$ for storing the result matrix C.
 */

#include "utils.h"
#include "cblas.h"


/**
 * @brief Computes C = A * B * B^T + A^T * A.
 * @param N Matrix dimension.
 * @param A Input matrix A.
 * @param B Input matrix B.
 * @return Pointer to resulting matrix C.
 */
double* my_solver(int N, double *A, double *B) {
	double* resultat = calloc(N * N, sizeof(double));
	double* resultat2 = calloc(N * N, sizeof(double));
	double* resultat3 = calloc(N * N, sizeof(double));
	double* resultat4 = calloc(N * N, sizeof(double));

	/**
	 * Block Logic: Iterative processing loop.
	 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
	 */
	for(int i = 0; i < N * N; i ++){
		resultat[i] = A[i];
	}

	cblas_dtrmm(CblasRowMajor, CblasLeft, CblasUpper, CblasTrans, CblasNonUnit, N, N, 1, A, N, resultat, N);

	/**
	 * Block Logic: Iterative processing loop.
	 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
	 */
	for(int i = 0; i < N * N; i ++){
		resultat2[i] = B[i];
	}

	cblas_dtrmm(CblasRowMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit, N, N, 1, A, N, resultat2, N);

	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, N, N, N, 1, resultat2, N, B, N, 1, resultat3, N);

	/**
	 * Block Logic: Iterative processing loop.
	 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
	 */
	for(int i = 0; i < N * N; i++){
    	resultat4[i] = resultat3[i] + resultat[i];
    }
	
    free(resultat3);
    free(resultat2);
    free(resultat);
    
	return resultat4;
}
