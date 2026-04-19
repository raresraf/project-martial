/**
 * @raw/8dfa30bd-f2c8-43eb-9acb-391166828a75/solver_opt.c
 * @brief Optimized dense matrix multiplication solver computing C = (A * B) * B^T + A^T * A.
 * Algorithm: Matrix multiplication employing pointer arithmetic and CPU registers for sum aggregation.
 * Time Complexity: $O(N^3)$
 * Space Complexity: $O(N^2)$ tracking internal partial product evaluations.
 */

#include "utils.h"

double* my_solver(int N, double *A, double* B) {
	register int i, j, k;
	register double *orig_p, *p1, *p2, sum;
	double *AB, *ABBt, *AtA, *C;

	/**
	 * Functional Utility: Initializes dynamically allocated matrix buffers and caches for intermediate outputs.
	 */
	AB = calloc(N * N, sizeof(double));
	if (AB == NULL)
		exit(-1);
	ABBt = calloc(N * N, sizeof(double));
	if (ABBt == NULL)
		exit(-1);	
	AtA = calloc(N * N, sizeof(double));
	if (AtA == NULL)
		exit(-1);	
	C = calloc(N * N, sizeof(double));
	if (C == NULL)
		exit(-1);

	/**
	 * Block Logic: Computes AB = A * B.
	 * Optimization: Employs explicit pointer tracking to evade multiplication overheads during indexing.
	 */
	for (i = 0; i < N; i++) {
		orig_p = A + i * N;
		for (j = 0; j < N; j++) {
			p1 = orig_p + i;
			p2 = &B[i * N + j];
			sum = 0.0;
			for (k = i; k < N; k++) {
				sum += *p1 * *p2;
				p1++;
				p2 += N;
			}
			AB[i * N + j] = sum;
		}
	}

	/**
	 * Block Logic: Computes ABBt = AB * B^T.
	 * Optimization: Accesses B in transposed layout through pointer iteration strategies.
	 */
	for (i = 0; i < N; i++) {
		orig_p = AB + i * N;
		for (j = 0; j < N; j++) {
			p1 = orig_p;
			p2 = &B[j * N];
			sum = 0.0;
			for (k = 0; k < N; k++) {
				sum += *p1 * *p2;
				p1++;
				p2++;
			}
			ABBt[i * N + j] = sum;
		}
	}

	/**
	 * Block Logic: Computes AtA = A^T * A.
	 * Optimization: Strides the internal pointer arrays vertically. Limits iterations structurally via `break`.
	 */
	for (i = 0; i < N; i++) {
		orig_p = A + i;
		for (j = 0; j < N; j++) {
			p1 = orig_p;
			p2 = &A[j];
			sum = 0.0;
			for (k = 0; k < N; k++) {
				sum += *p1 * *p2;
				p1 += N;
				p2 += N;
				if (k == i || k == j)
					break;
			}
			AtA[i * N + j] = sum;
		}
	}

	/**
	 * Block Logic: Synthesizes final summation C = ABBt + AtA.
	 * Invariant: Aggregates computed matrices sequentially using predefined loops.
	 */
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			C[i * N + j] = ABBt[i * N + j] + AtA[i * N + j];
		}
	}

	free(AB);
	free(ABBt);
	free(AtA);

	return C;	
}
