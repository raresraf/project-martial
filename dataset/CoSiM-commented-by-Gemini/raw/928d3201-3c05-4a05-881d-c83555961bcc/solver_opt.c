/**
 * @raw/928d3201-3c05-4a05-881d-c83555961bcc/solver_opt.c
 * @brief Optimized dense matrix multiplication solver computing C = A * (B * B^T) + A^T * A.
 * Algorithm: Improves memory access latency by precomputing transposed loops and utilizing pointer dereferencing for multiplication.
 * Time Complexity: $O(N^3)$
 * Space Complexity: $O(N^2)$ due to auxiliary storage required for transposed layout buffers.
 */

#include "utils.h"

double* my_solver(int N, double *A, double* B) {
	printf("OPT SOLVER\n");

	register int i, j, k;
	register double aux;
	
	/**
	 * Functional Utility: Allocates block memory for intermediate products.
	 */
	double *BBt = (double*) calloc(N * N, sizeof(double));

	/**
	 * Block Logic: Evaluates BBt = B * B^T using local scope variables and direct memory access.
	 * Optimization: Eliminates repetitive coordinate multiplication addressing inside deep loops.
	 */
	for (i = 0; i < N; ++i) {
		register double *orig_pb = &B[i * N + 0];
		for (j = 0; j < N; ++j) {
			register double *pb = orig_pb;
			register double *pb_t = &B[j * N + 0];
			aux = 0.0;

			for (k = 0; k < N; ++k) {
				aux += *pb * *pb_t;
				++pb;
				++pb_t;
			}

			BBt[i * N + j] = aux;
		}
	}

	/**
	 * Functional Utility: Configures target accumulator ABBt.
	 */
	double *ABBt = (double*) calloc(N * N, sizeof(double));

	/**
	 * Block Logic: Executes A * BBt relying on localized variable pointer traversals.
	 * Optimization: Uses cache efficient striding scaling up through `N` limits.
	 */
	for (i = 0; i < N; ++i) {
		register double *orig_pa = &A[i * N + i];
		for (j = 0; j < N; ++j) {
			register double *pa = orig_pa;
			register double *pbbt = &BBt[i * N + j];
			aux = 0.0;

			for (k = i; k < N; ++k) {
				aux += *pa * *pbbt;
				++pa;
				pbbt += N;
			}

			ABBt[i * N + j] = aux;
		}
	}

	free(BBt);

	/**
	 * Functional Utility: Allocates memory representing the final symmetric AAt component.
	 */
	double *AAt = (double*) calloc(N * N, sizeof(double));

	/**
	 * Block Logic: Operates sequentially upon the structural confines producing A^T * A.
	 * Optimization: Condenses loops bounded functionally to eliminate evaluating pre-defined zeroes.
	 */
	for (i = 0; i < N; ++i) {
		register double *orig_pa = &A[i];
		for (j = 0; j < N; ++j) {
			register double *pa = orig_pa;
			register double *pa_t = &A[j];
			aux = 0.0;

			for (k = 0; k <= ((i < j) ? i : j); ++k) {
				aux += *pa * *pa_t;
				pa += N;
				pa_t += N;
			}

			AAt[i * N + j] = aux;
		}
	}

	double *res = (double*) calloc(N * N, sizeof(double));

	/**
	 * Block Logic: Single dimension loop summing precalculated output elements.
	 * Invariant: Executes an identical map between identical array forms res = ABBt + AAt.
	 */
	for (i = 0; i < N * N; ++i) {
		res[i] = ABBt[i] + AAt[i];
	}

	free(ABBt);
	free(AAt);

	return res;
}
