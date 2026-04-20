/**
 * @file solver_blas.c
 * @brief BLAS-based optimized matrix solver computing $C = A \times B \times B^T + A^T \times A$.
 * Algorithm: Relies on optimized numerical linear algebra kernels (DGEMM, DTRMM).
 * Time Complexity: $O(N^3)$ due to dense matrix multiplications.
 * Space Complexity: $O(N^2)$ for storing the result matrix C.
 */

#include "utils.h"
#include "cblas.h"
#include <string.h>


#define DIE(assertion, call_description)				\
	do {								\
		/**
		 * Block Logic: Conditional state branch.
		 * Invariant: The conditional branch maintains control flow invariants.
		 */
		if (assertion) {					\
			fprintf(stderr, "(%s, %d): ",			\
					__FILE__, __LINE__);		\
			perror(call_description);			\
			exit(EXIT_FAILURE);				\
		}							\
	} while (0)




/**
 * @brief Computes C = A * B * B^T + A^T * A.
 * @param N Matrix dimension.
 * @param A Input matrix A.
 * @param B Input matrix B.
 * @return Pointer to resulting matrix C.
 */
double* my_solver(int N, double *A, double *B) {
	double *M;
	double *AB;
	double *ATA;
	double *ABBT;
	int i, j;

	M = (double *)malloc(N * N * sizeof(double));
	DIE (M == NULL, "malloc M");

	AB = (double *)malloc(N * N * sizeof(double));
	DIE(AB == NULL, "malloc AB");

	ABBT = (double *)malloc(N * N * sizeof(double));
	DIE(AB == NULL, "malloc ABBT");

	ATA = (double *)malloc(N * N * sizeof(double));
	DIE(ATA == NULL, "malloc ATA");

	
	memcpy(AB, B, N * N * sizeof(double));
	cblas_dtrmm(CblasRowMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit,
				N, N, 1.0, A, N, AB, N);
	
	
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, N, N, N, 1, AB, N,
				B, N, 0, ABBT, N);
	
	
	memcpy(ATA, A, N * N * sizeof(double));
	cblas_dtrmm(CblasRowMajor, CblasLeft, CblasUpper, CblasTrans, CblasNonUnit,
				N, N, 1.0, A, N, ATA, N);
	
	
	/**
	 * Block Logic: Iterative processing loop.
	 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
	 */
	for (i = 0; i < N; i++) {
		/**
		 * Block Logic: Iterative processing loop.
		 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
		 */
		for (j = 0; j < N; j++) {
			M[i * N + j] = ATA[i * N + j] + ABBT[i * N + j];
		}
	}

	free(ATA);
	free(ABBT);
	free(AB);

	return M;
}
