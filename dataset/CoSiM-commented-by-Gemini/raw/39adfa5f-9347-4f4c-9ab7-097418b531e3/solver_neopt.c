/**
 * @file solver_neopt.c
 * @brief Unoptimized implementation of matrix solver.
 *
 * Computes A * B * B^T + A^T * A. Focuses on straightforward translation
 * of the mathematical operations without considering performance optimization.
 */

#include "utils.h"

/**
 * @brief Solves the matrix equation A * B * B^T + A^T * A.
 *
 * @param N Matrix dimension.
 * @param A Pointer to the first input matrix (assumed upper triangular).
 * @param B Pointer to the second input matrix.
 * @return Pointer to the resulting matrix, or NULL on allocation failure.
 */
double* my_solver(int N, double *A, double* B) {
	printf("NEOPT SOLVER\n");

    
	double *BBt = (double *)calloc(N * N, sizeof(double));
	/**
	 * @brief Computes B * B^T.
	 * Pre-condition: B is fully initialized.
	 * Invariant: BBt stores partial dot products up to row i, column j.
	 */
	for (int i = 0; i < N; i++) {
		/**
		 * @brief Process columns of the resulting BBt matrix.
		 * Pre-condition: Valid row index i.
		 * Invariant: BBt[i, j] accumulates the product.
		 */
		for (int j = 0; j < N; j++) {
			/**
			 * @brief Compute the dot product of B row i and B row j.
			 * Pre-condition: Valid row and column indices.
			 * Invariant: Accumulates multiplication element by element.
			 */
			for (int k = 0; k < N; k++) {
				BBt[i * N + j] += B[i * N + k] * B[j * N + k];
			}
		}
	}

    
    double *A_BBt = (double *)calloc(N * N, sizeof(double));
    /**
	 * @brief Computes A * (B * B^T).
	 * Pre-condition: BBt is fully computed; A is upper triangular.
	 * Invariant: A_BBt stores partial dot products up to row i, column j.
	 */
    for (int i = 0; i < N; i++) {
		/**
		 * @brief Process columns of the resulting A_BBt matrix.
		 * Pre-condition: Valid row index i.
		 * Invariant: A_BBt[i, j] accumulates the product.
		 */
		for (int j = 0; j < N; j++) {
			/**
			 * @brief Compute the dot product utilizing A's upper triangular form.
			 * Pre-condition: A[i, k] is zero for k < i.
			 * Invariant: Accumulates non-zero multiplications.
			 */
			for (int k = i; k < N; k++) {
				A_BBt[i * N + j] += A[i * N + k] * BBt[k * N + j];
			}
		}
	}

    
    double *AtA = (double *)calloc(N * N, sizeof(double));
    /**
	 * @brief Computes A^T * A.
	 * Pre-condition: A is upper triangular.
	 * Invariant: AtA stores partial dot products up to row i, column j.
	 */
    for (int i = 0; i < N; i++) {
		/**
		 * @brief Process columns of the resulting AtA matrix.
		 * Pre-condition: Valid row index i.
		 * Invariant: AtA[i, j] accumulates the product.
		 */
		for (int j = 0; j < N; j++) {
			/**
			 * @brief Compute dot product utilizing A's upper triangular form for both A^T and A.
			 * Pre-condition: A^T row i has non-zeros up to i.
			 * Invariant: Accumulates non-zero multiplications.
			 */
			for (int k = 0; k <= i; k++) {
				AtA[i * N + j] += A[k * N + i] * A[k * N + j];
			}
		}
	}

    
    /**
	 * @brief Adds AtA to A_BBt to form the final result.
	 * Pre-condition: Both A_BBt and AtA are fully computed.
	 * Invariant: A_BBt accumulates the sum element by element.
	 */
    for (int i = 0; i < N; i++) {
		/**
		 * @brief Process columns for addition.
		 * Pre-condition: Valid row index i.
		 * Invariant: Elements up to j are summed.
		 */
		for (int j = 0; j < N; j++) {
			A_BBt[i * N + j] += AtA[i * N + j];
		}
	}

	
    free(AtA);
    free(BBt);
	return A_BBt;
}
