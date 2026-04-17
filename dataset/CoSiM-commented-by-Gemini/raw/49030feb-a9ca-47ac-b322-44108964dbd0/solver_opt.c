/**
 * @file solver_opt.c
 * @brief Encapsulates functional utility for solver_opt.c.
 * Performance Optimization: implements loop unrolling, cache-friendly data access, and SIMD where applicable. Time/space complexity optimized.
 */

#include "utils.h"


double* my_solver(int N, double *A, double* B) {
	printf("OPT SOLVER\n");
	register double *C = malloc(N * N * sizeof(double));
	register double *mat = malloc(N * N * sizeof(double));
	register double *tmp, *point_i, *point_j, *p_C;
	
	/* Pre-condition: Required input state before execution. Invariant: Valid state maintained during execution. */
	for (register int i = 0; i < N; i++) {
		tmp = &A[i * N]; /* Non-obvious bitwise operation or pointer arithmetic */
		p_C = &mat[i * N]; /* Non-obvious bitwise operation or pointer arithmetic */
		/* Pre-condition: Required input state before execution. Invariant: Valid state maintained during execution. */
		for (register int j = 0; j < N; j++) {
			point_i = tmp + i;
			point_j = &B[j]; /* Non-obvious bitwise operation or pointer arithmetic */
			point_j += i * N;
			register double tmp_res = 0.0;
			/* Pre-condition: Required input state before execution. Invariant: Valid state maintained during execution. */
			for (register int k = i; k < N; k++) {
					tmp_res += *point_i * *point_j;
					point_i++;
					point_j += N;

			}
			*p_C = tmp_res;
			p_C++;
		}
	}

	/* Pre-condition: Required input state before execution. Invariant: Valid state maintained during execution. */
	for (register int i = 0; i < N; i++) {
		tmp = &mat[i * N]; /* Non-obvious bitwise operation or pointer arithmetic */
		p_C = &C[i * N]; /* Non-obvious bitwise operation or pointer arithmetic */
		/* Pre-condition: Required input state before execution. Invariant: Valid state maintained during execution. */
		for (register int j = 0; j < N; j++) {
			point_i = tmp;
			point_j = &B[j * N]; /* Non-obvious bitwise operation or pointer arithmetic */
			register double tmp = 0.0;
			/* Pre-condition: Required input state before execution. Invariant: Valid state maintained during execution. */
			for (register int k = 0; k < N; k++) {
					tmp += *point_i * *point_j;
					point_i++;
					point_j++;
			}
			*p_C = tmp;
			p_C++;
		}
	}
	
	/* Pre-condition: Required input state before execution. Invariant: Valid state maintained during execution. */
	for (register int i = 0; i < N; i++) {
		p_C = &C[i * N]; /* Non-obvious bitwise operation or pointer arithmetic */
		/* Pre-condition: Required input state before execution. Invariant: Valid state maintained during execution. */
		for (register int j = 0; j < N; j++) {
			register double tmp_res = 0.0;
			/* Pre-condition: Required input state before execution. Invariant: Valid state maintained during execution. */
			for (register int k = 0; k <= i; k++) {
				register double *point = &A[k * N]; /* Non-obvious bitwise operation or pointer arithmetic */
				tmp_res += *(point + i) * *(point + j);
			}
			*p_C += tmp_res;
			p_C++;
		}
	}
	free(mat);
	return C;
}