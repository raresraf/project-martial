/**
 * @raw/ab6ed51d-ae33-48f2-bee3-efa7e7895107/solver_opt.c
 * @brief Optimized dense matrix multiplication solver computing C = (A * B) * B^T + A^T * A.
 * Algorithm: Improves loop ordering taking cache latency constraints into consideration by utilizing explicitly tracked variables.
 * Time Complexity: $O(N^3)$
 * Space Complexity: $O(N^2)$ due to auxiliary storage required for distinct buffers mapping segments.
 */

#include "utils.h"
#include <string.h>

double* my_solver(int N, double *A, double* B) {
	printf("OPT SOLVER\n");

	register int i,j,k;
	
	/**
	 * Functional Utility: Provisions intermediary matrices enforcing clean bounds tracking elements.
	 */
	double *C = malloc(sizeof(double) * N * N);
	if (!C) 
		return NULL;

	double *AB = malloc(sizeof(double) * N * N);
	if (!AB)
		return NULL;

	double *BBt = malloc(sizeof(double) * N * N);
	if (!BBt)
		return NULL;

	double *AtA = malloc(sizeof(double) * N * N);
	if (!AtA)
		return NULL;

	/**
	 * Block Logic: Generates AB = A * B mapping index strides locally using predefined array elements.
	 * Optimization: Decreases execution footprint substituting indexing steps with additive pointer bounds.
	 */
	for (i = 0; i < N; i++) {
		register double	*a = A + i * N;
		for (j = 0; j < N; j++) {
			register double *aa = a + i;
			register double *b = B + j + i * N;
			register double	sum = 0.0;
			for (k = i; k < N; k++) {
				sum += *aa * *b;
				aa++;
				b += N;
			}
			AB[i * N + j] = sum;
		}
	}

	/**
	 * Block Logic: Evaluates the sub-component producing BBt = AB * B^T effectively resolving (A * B) * B^T.
	 * Optimization: Sequences cache allocations extracting identical constants externally to mitigate inner processing operations.
	 */
	for (i = 0; i < N; i++) {
		register double	*a = AB + i * N;
		for (j = 0; j < N; j++) {
			register double	*aa = a;
			register double	*b = B + j * N;
			register double	sum = 0.0;
			for (k = 0; k < N; k++) {
				sum += *aa * *b;
				aa++;
				b++;
			}
			BBt[i * N + j] = sum;
		}
	}

	/**
	 * Block Logic: Solves AtA = A^T * A.
	 * Optimization: Extends local memory optimization logic avoiding sequential recalculation delays defining arrays individually inside execution.
	 */
	for (i = 0; i < N; i++) {
		register double	*a = A + i;
		for (j = 0; j < N; j++) {
			register double	sum = 0.0;
			for (k = 0; k < N; k++) {
				register double	*aa = a + k * N;
				register double	*b = A + j + k * N;
				sum += *aa * *b;
				aa+=N;
				b+=N;
			}
			AtA[i * N + j] = sum;
		}
	}

	/**
	 * Block Logic: Solves the equation consolidating previously completed structures rendering final vector inside C.
	 * Invariant: Executes sequential iterations traversing contiguous cache elements linearly mitigating spatial cache misses.
	 */
	for (i = 0; i < N; i++) {
		register double	*a = BBt + i * N;
		register double	*aa = AtA + i * N;
		for (j = 0; j < N; j++) {
			C[i * N + j] = *aa + *a;
			aa++;
			a++;
		}
	}

	free(BBt);
	free(AtA);
	free(AB);

	return C;	
}
