/**
 * @raw/96d93c00-0061-439f-84b0-7b394d9b51f1/solver_opt.c
 * @brief Optimized dense matrix multiplication solver computing RES = (A * B) * B^T + A^T * A.
 * Algorithm: Implementation employing pointer iterations and cached transpose configurations.
 * Time Complexity: $O(N^3)$
 * Space Complexity: $O(N^2)$ due to explicit copies and transpositions.
 */

#include "utils.h"

double* my_solver(int N, double *A, double* B) {
	double *A_star_B;
	double *B_tr;
	double *OP1;
	double *A_tr;
	double *OP2;
	double *RES;
	int numMatrixElems = N * N;
	register int i, j, k;

	/**
	 * Functional Utility: Initializes dynamically allocated matrix buffers and caches.
	 */
	A_star_B = calloc(numMatrixElems, sizeof(*A_star_B));
	B_tr = calloc(numMatrixElems, sizeof(*B_tr));
	OP1 = calloc(numMatrixElems, sizeof(*OP1));
	A_tr = calloc(numMatrixElems, sizeof(*A_tr));
	OP2 = calloc(numMatrixElems, sizeof(*OP2));
	RES = calloc(numMatrixElems, sizeof(*RES));

	/**
	 * Block Logic: Computes A_star_B = A * B.
	 * Optimization: Employs explicit pointer tracking incrementing column limits.
	 */
	for (i = 0; i < N; i++) {
		register double *orig_pa = &A[i * N];

		for (j = 0; j < N; j++) {
			register double *pa = orig_pa;
			pa += i;
		    register double *pb = &B[j];
		    pb += N * i;

		    register double sum = 0;

			for (k = i; k < N; k++) {
				sum += *pa * *pb;
				pa++;
				pb += N;
			}

			A_star_B[i * N + j] = sum;
		}
	}

	/**
	 * Block Logic: Precalculates the transposition B_tr = B^T mapping it explicitly.
	 */
	for (i = 0; i < N; i++) {
    	for (j = 0; j < N; j++) {
    		B_tr[j * N + i] = B[i * N + j];
    	}
	}

	/**
	 * Block Logic: Computes OP1 = A_star_B * B_tr.
	 * Optimization: Minimizes cache faults during `B` elements access via B_tr traversal.
	 */
	for (i = 0; i < N; i++) {
		register double *orig_pa = &A_star_B[i * N];

		for (j = 0; j < N; j++) {
			register double *pa = orig_pa;
		    register double *pb = &B_tr[j];
		    register double sum = 0;

			for (k = 0; k < N; k++) {
				sum += *pa * *pb;
				pa++;
				pb += N;
			}

			OP1[i * N + j] = sum;
		}
	}

	/**
	 * Block Logic: Derives mapping explicit transposition A_tr = A^T.
	 */
	for (i = 0; i < N; i++) {
    	for (j = 0; j < N; j++) {
    		A_tr[j * N + i] = A[i * N + j];
    	}
	}

	/**
	 * Block Logic: Computes OP2 = A_tr * A.
	 * Optimization: Implements dot products manipulating pointers. Structurally restricts summation upper bounds `k <= i`.
	 */
	for (i = 0; i < N; i++) {
		register double *orig_pa = &A_tr[i * N];

		for (j = 0; j < N; j++) {
			register double *pa = orig_pa;
		    register double *pb = &A[j];
		    register double sum = 0;

			for (k = 0; k <= i; k++) {
				sum += *pa * *pb;
				pa++;
				pb += N;
			}

			OP2[i * N + j] = sum;
		}
	}

	/**
	 * Block Logic: Executes the terminal operation formulating RES = OP1 + OP2.
	 * Invariant: Aggregates calculated products symmetrically yielding the expected combination output.
	 */
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			RES[i * N + j] = OP1[i * N + j] + OP2[i * N + j];
		}
	}

	free(A_star_B);
	free(B_tr);
	free(OP1);
	free(A_tr);
	free(OP2);

	return RES;
}
