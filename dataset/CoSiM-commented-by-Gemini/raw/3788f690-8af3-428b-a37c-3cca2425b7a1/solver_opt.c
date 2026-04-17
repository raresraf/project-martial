/**
 * @file solver_opt.c
 * @brief Optimized implementation of matrix operations utilizing memory access patterns.
 *
 * Implements the mathematical formula C = A * B * B^T + A^T * A.
 * Optimizations include pointer arithmetic, register variables, and loop ordering
 * for improved cache locality and execution speed.
 */

#include <string.h>
#include "utils.h"

/**
 * @brief Solves the matrix equation C = A * B * B^T + A^T * A with optimizations.
 *
 * @param N Matrix dimension (N x N).
 * @param A Pointer to the first input matrix (assumed upper triangular).
 * @param B Pointer to the second input matrix.
 * @return Pointer to the resulting matrix C, or NULL on allocation failure.
 */
double* my_solver(int N, double *A, double* B) {
	double *C = (double*) calloc(N * N, sizeof(double));
	if (!C)
		return NULL;

	double *aux = (double*) calloc(N * N, sizeof(double));
	if (!aux)
		return NULL;
	int i, j, k;
	/**
	 * @brief Computes aux = A * B using pointer arithmetic and register variables.
	 * Pre-condition: A is an upper triangular matrix.
	 * Invariant: aux stores computed rows up to index i.
	 */
	for (i = 0; i < N; i++) {
		double *orig_pa = &A[i * N];
		/**
		 * @brief Process columns of B while traversing rows of A.
		 * Pre-condition: orig_pa points to the start of row i in A.
		 * Invariant: aux[i, j] accumulates the precise dot product.
		 */
		for (j = 0; j < N; j++) {
			register double *pa = orig_pa + i;
			register double *pb = &B[i * N + j];
			register double suma = 0;
			/**
			 * @brief Inner loop optimized with continuous memory access via pointers.
			 * Pre-condition: Pointer references start from the main diagonal of A.
			 * Invariant: suma maintains the running total of the dot product.
			 */
			for (k = i; k < N; k++) {
				suma += *pa * *pb;
				pa++;
				pb += N;
			}
			aux[i * N + j] = suma;
		}
	}

	/**
	 * @brief Computes C = aux * B^T using cache-friendly transposed traversal.
	 * Pre-condition: aux contains the intermediate result A * B.
	 * Invariant: C stores computed rows up to index i.
	 */
	for (i = 0; i < N; i++) {
		double *origin_aux = &aux[i * N];
		/**
		 * @brief Traverse rows of B (acting as columns of B^T) to exploit memory locality.
		 * Pre-condition: origin_aux points to the start of row i in aux.
		 * Invariant: C[i, j] accumulates the precise dot product.
		 */
		for (j = 0; j < N; j++) {
			register double suma = 0;
			register double *paux = origin_aux;
			register double *pb_t = &B[j *  N];
			/**
			 * @brief Inner loop optimized for sequential memory access.
			 * Pre-condition: Pointers advance linearly through respective rows.
			 * Invariant: suma maintains the running total of the dot product.
			 */
			for (k = 0; k < N; k++) {
				suma += *paux * *pb_t;
				paux++;
				pb_t ++;
			}
			C[i * N + j] = suma;
		}
	}

	memset(aux, 0, N * N * sizeof(double));
	/**
	 * @brief Computes aux = A^T * A.
	 * Pre-condition: aux buffer is cleared.
	 * Invariant: aux stores computed rows up to index i.
	 */
	for (i = 0; i < N; i++) {
		double *origin_at = &A[i];
		/**
		 * @brief Process inner iterations constrained by the upper triangular structure of A.
		 * Pre-condition: origin_at points to column i elements representing row i of A^T.
		 * Invariant: aux[i, j] accumulates the dot product.
		 */
		for (j = 0; j < N; j++) {
			register double suma = 0;
			register double *pa_t = origin_at;
			register double *pa = &A[j];
			/**
			 * @brief Optimized partial column traversal to match transposed upper triangular bounds.
			 * Pre-condition: Pointers jump down columns (stride N).
			 * Invariant: suma maintains the running total.
			 */
			for (k = 0; k < i + 1; k++) {
				suma += *pa_t * *pa;
				pa_t+= N;
				pa += N;
			}
			aux[i * N + j] = suma;
		}
	}

	/**
	 * @brief Accumulate final result C = C + aux.
	 * Pre-condition: Matrices C and aux are fully computed.
	 * Invariant: Rows up to index i are summed.
	 */
	for (i = 0; i < N; i++) {
		/**
		 * @brief Element-wise addition.
		 * Pre-condition: Valid row index i.
		 * Invariant: Elements up to j are summed.
		 */
		for (j = 0; j < N; j++) {
			C[i * N + j] += aux[i * N + j];
		}
	}

	free(aux);

	return C;
}
