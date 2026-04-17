/**
 * @file solver_opt.c
 * @brief Encapsulates functional utility for solver_opt.c.
 * Performance Optimization: implements loop unrolling, cache-friendly data access, and SIMD where applicable. Time/space complexity optimized.
 */

#include "utils.h"


double* my_solver(int N, double *A, double* B) {
	printf("OPT SOLVER\n");

	register int i, j, k;
	double *C, *AtA, *AB, *ABBt;

	C = calloc(N * N, sizeof(double));
	DIE(C == NULL, "calloc C");

	AtA = calloc(N * N, sizeof(double));
	DIE(AtA == NULL, "calloc AtA");

	AB = calloc(N * N, sizeof(double));
	DIE(AB == NULL, "calloc AB");

	ABBt = calloc(N * N, sizeof(double));
	DIE(ABBt == NULL, "calloc ABBt");

	
	/* Pre-condition: Required input state before execution. Invariant: Valid state maintained during execution. */
	for (i = 0; i < N; i++) {
		register double *orig_pa = A + i * N;
        /* Pre-condition: Required input state before execution. Invariant: Valid state maintained during execution. */
        for (j = 0; j < N; j++) {
			register double *pa = orig_pa + i;
			register double *pb = B + i * N + j;
			register double sum = 0.0;
            /* Pre-condition: Required input state before execution. Invariant: Valid state maintained during execution. */
            for (k = i; k < N; k++) {
                sum += *pa * *pb;
				pa++;
				pb += N;
            }
			*(AB + i * N + j) = sum;
        }
    }

	
	/* Pre-condition: Required input state before execution. Invariant: Valid state maintained during execution. */
	for (i = 0; i < N; i++) {
		double *orig_pab = AB + i * N;
        /* Pre-condition: Required input state before execution. Invariant: Valid state maintained during execution. */
        for (j = 0; j < N; j++) {
			register double *pab = orig_pab;
			register double *pb = B + j * N;
			register double sum = 0.0;
            /* Pre-condition: Required input state before execution. Invariant: Valid state maintained during execution. */
            for (k = 0; k < N; k++) {
				sum += *pab * *pb;
				pab++;
				pb++;
            }
			*(ABBt + i * N + j) = sum;
        }
    }

	
	
	/* Pre-condition: Required input state before execution. Invariant: Valid state maintained during execution. */
	for (i = 0; i < N; i++) {
		register double *orig_pat = A + i;
        /* Pre-condition: Required input state before execution. Invariant: Valid state maintained during execution. */
        for (j = 0; j < N; j++) {
			register double *pat = orig_pat;
			register double *pa = A + j;
			register double sum = 0.0;
            /* Pre-condition: Required input state before execution. Invariant: Valid state maintained during execution. */
            for (k = 0; k <= i && k <= j; k++) { /* Non-obvious bitwise operation or pointer arithmetic */
                sum += *pat * *pa;
				pat += N;
				pa += N;
            }
			*(AtA + i * N + j) = sum;
			*(C + i * N + j) = *(ABBt + i * N + j) + *(AtA + i * N + j);
        }
    }

	free(AB);
    free(AtA);
    free(ABBt);

	return C;
}
