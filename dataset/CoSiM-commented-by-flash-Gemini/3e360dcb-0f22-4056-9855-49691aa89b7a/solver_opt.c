
#include "utils.h"

/**
 * @file solver_opt.c
 * @brief Manually optimized matrix solver implementation.
 * 
 * Functional Intent: Computes the resulting matrix for the expression:
 * Result = (A^T * A) + A * (B * B^T)
 * where A is an upper triangular matrix.
 * 
 * Performance Optimization: Employs several manual C tuning techniques:
 * 1. Explicit Transposition: Pre-computes A^T and B^T to convert strided 
 *    column access into sequential row access, significantly improving 
 *    spatial locality in the innermost loops.
 * 2. Pointer Arithmetic: Uses direct pointer increments instead of array 
 *    indexing to reduce address calculation overhead.
 * 3. Register Caching: Keeps loop counters and frequently used pointers 
 *    in CPU registers to minimize memory latency.
 * 4. Triangular Optimization: Restricts dot-product ranges for matrix A.
 * 
 * Domain: HPC, Linear Algebra, Manual Performance Tuning.
 */

/**
 * my_solver - Optimized matrix solver with pre-transposition and pointer-based loops.
 */
double* my_solver(int N, double *A, double *B) {
	double *C;
	double *AtA, *BBt, *ABBt;
	double *At, *Bt;
	register int i;
	register int j;
	register int k;

	register int size = N * N * sizeof(*C);
	C = malloc(size);
	AtA = malloc(size);
	At = malloc(size);
	Bt = malloc(size);
	BBt = malloc(size);
	ABBt = malloc(size);
	if (C == NULL || AtA == NULL || BBt == NULL || ABBt == NULL) {
		exit(EXIT_FAILURE);
	}

	/**
	 * Block Logic: Pre-transposition phase.
	 * Optimization: Reorders memory for A and B to enable cache-friendly sequential 
	 * access in subsequent multiplication phases.
	 */
	for (i = 0; i < N; ++i) {
		register double *At_col = At + i;
		register double *Bt_col = Bt + i;
		register double *pA = A + i * N;
		register double *pB = B + i * N;

		for (j = 0; j < N; ++j) {
			*At_col = *pA;
			*Bt_col = *pB;
			At_col += N;
			Bt_col += N;
			++pA;
			++pB;
		}
	}

	/**
	 * Block Logic: Compute AtA = A^T * A.
	 * Optimization: Uses pre-computed At to enable sequential pointer traversal 
	 * (pat++) for the first operand.
	 */
	register double *AtA_ptr = AtA;
	for (i = 0; i < N; ++i) {
		register double *orig_pat = At + i * N;
		for (j = 0; j < N; ++j) {
			register double *pat = orig_pat;
			register double *pa = A + j;
			register double suma = 0.0;
			for (k = 0; k <= i; ++k) {
				suma += *pat * *pa;
				++pat;
				pa += N;
			}
			*AtA_ptr = suma;
			++AtA_ptr;
		}
	}

	/**
	 * Block Logic: Compute BBt = B * B^T.
	 * Optimization: Dual sequential access using pre-computed Bt.
	 */
	register double *BBt_ptr = BBt;
	for (i = 0; i < N; ++i) {
		register double *orig_pb = B + i * N;
		for (j = 0; j < N; ++j) {
			register double *pb = orig_pb;
			register double *pbt = Bt + j;
			register double suma = 0.0;
			for (k = 0; k < N; ++k) {
				suma += *pb * *pbt;
				++pb;
				pbt += N;
			}
			*BBt_ptr = suma;
			++BBt_ptr;
		}
	}

	/**
	 * Block Logic: Compute ABBt = A * BBt.
	 * Optimization: Triangular pointer increments.
	 * Logic: Starts k from i to skip zeroes in upper-triangular matrix A.
	 */
	register double *ABBt_ptr = ABBt;
	for (i = 0; i < N; ++i) {
		register double *orig_pa = A + i * N;
		for (j = 0; j < N; ++j) {
			register double *pa = orig_pa + i;
			register double *pbbt = BBt + i * N + j;
			register double suma = 0.0;
			for (k = i; k < N; ++k) {
				suma += *pa * *pbbt;
				++pa;
				pbbt += N;
			}
			*ABBt_ptr = suma;
			++ABBt_ptr;
		}
	}

	/**
	 * Block Logic: Final aggregation.
	 */
	for (i = 0; i < N; ++i) {
		for (j = 0; j < N; ++j) {
			*(C + i * N + j) = *(ABBt + i * N + j) + *(AtA + i * N + j);
		}
	}

	free(ABBt);
	free(AtA);
	free(BBt);
	free(At);
	free(Bt);
	return C;	
}
