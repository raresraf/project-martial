/**
 * @raw/925c9aec-7fc6-4d6b-aed5-ede8c01dc703/solver_neopt.c
 * @brief Unoptimized dense matrix multiplication solver computing C = A * B * B^T + A^T * A.
 * Algorithm: Standard nested loops generating explicit transposition maps and partial multiplications.
 * Time Complexity: $O(N^3)$
 * Space Complexity: $O(N^2)$ representing intermediate target buffers.
 */

#include "utils.h"

double* my_solver(int N, double *A, double* B) {
	double *BBT, *ABBT, *ATA, *C, *AT;
	
	/**
	 * Functional Utility: Initializes dynamically allocated matrix buffers and caches for intermediate outputs.
	 */
	BBT = calloc(N * N , sizeof(double));
	ABBT = calloc(N * N , sizeof(double));
	ATA = calloc(N * N , sizeof(double));
	AT = calloc(N * N, sizeof(double));
	C = calloc(N * N, sizeof(double));

	/**
	 * Block Logic: Computes BBT = B * B^T.
	 * Invariant: Exploits symmetry of the resulting matrix; iterates across the upper triangle and mirrors it.
	 */
    for (int i = 0; i < N; ++i) {
      for (int j = i; j < N; ++j) {
         for (int k = 0; k < N; ++k) {
            BBT[j * N + i] = BBT[i * N + j] += B[i * N + k] * B[j * N + k];
         }
      }
   }

	/**
	 * Block Logic: Computes ABBT = A * (B * B^T).
	 * Invariant: Applies upper-triangular constraints of A (where `k` begins at `i`).
	 */
    for (int i = 0; i < N; ++i) {
      for (int j = 0; j < N; ++j) {
         for (int k = i; k < N; ++k) {
            ABBT[i * N + j] += A[i * N + k] * BBT[k * N + j];
         }
      }
   }

	/**
	 * Block Logic: Pre-computes transpose of A to optimize subsequent multiplication cache hits.
	 * Invariant: Populates AT = A^T iteratively mapping contiguous memory.
	 */
    for (register int i = 0; i < N; i++) {
        for (register int j = 0; j <= i ; j++) {
            AT[i * N + j] = A[j * N + i];
		}
	}
   
	/**
	 * Block Logic: Computes ATA = A^T * A.
	 * Invariant: Matrix symmetric property computation restricted across the upper triangular half.
	 */
    for (int i = 0; i < N; ++i) {
		for (int j = i; j < N; ++j) {
			for (int k = 0; k < N; ++k) {
				ATA[i * N + j] += A[k * N + i] * A[k * N + j];
			}
		}
   }

	/**
	 * Block Logic: Final summation generating C = ABBT + ATA.
	 * Invariant: Overlays the symmetric elements simultaneously to minimize memory traversals.
	 */
    for (register int i = 0; i < N; ++i) {
    	for (register int j = i; j < N; ++j) {
			C[i * N + j] = ABBT[i * N + j] + ATA[i * N + j];
			C[j * N + i] = ABBT[j * N + i] + ATA[i * N + j];
		}
	}

	free(BBT);
	free(ABBT);
	free(AT);
	free(ATA);

	return C;
}
