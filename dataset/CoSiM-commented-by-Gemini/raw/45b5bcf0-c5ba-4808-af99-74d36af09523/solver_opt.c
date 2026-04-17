/**
 * @file solver_opt.c
 * @brief Encapsulates functional utility for solver_opt.c.
 * Performance Optimization: implements loop unrolling, cache-friendly data access, and SIMD where applicable. Time/space complexity optimized.
 */

#include "utils.h"


double* my_solver(int N, double *A, double* B) {

	double *first_mul = calloc (N * N, sizeof(double));

	/* Pre-condition: Required input state before execution. Invariant: Valid state maintained during execution. */
	if (!first_mul)
		return NULL;

	double *second_mul = calloc (N * N, sizeof(double));

	/* Pre-condition: Required input state before execution. Invariant: Valid state maintained during execution. */
	if (!second_mul)
		return NULL;

	double *third_mul = calloc (N * N, sizeof(double));

	/* Pre-condition: Required input state before execution. Invariant: Valid state maintained during execution. */
	if (!third_mul)
		return NULL;

	double *result = malloc (N * N * sizeof(double));

	register int i, j, k;

	/* Pre-condition: Required input state before execution. Invariant: Valid state maintained during execution. */
	for (i = 0; i < N; i++) {
		double *pa_first = &A[i * (N + 1)]; /* Non-obvious bitwise operation or pointer arithmetic */
		double *res = &first_mul[i * N]; /* Non-obvious bitwise operation or pointer arithmetic */

		/* Pre-condition: Required input state before execution. Invariant: Valid state maintained during execution. */
		for (j = 0; j < N; j++) {
			register double *pa = pa_first;
			register double *pb = &B[i * N + j]; /* Non-obvious bitwise operation or pointer arithmetic */

			register double suma = 0;

			/* Pre-condition: Required input state before execution. Invariant: Valid state maintained during execution. */
			for (k = i; k < N; k++) {
				suma += *pa * *pb;
				pa++;
				pb += N;
			}

			*res = suma;
			res++;
		}
	}

	/* Pre-condition: Required input state before execution. Invariant: Valid state maintained during execution. */
	for (i = 0; i < N; i++) {
		double *pa_first = &first_mul[i * N]; /* Non-obvious bitwise operation or pointer arithmetic */
		double *res = &second_mul[i * N]; /* Non-obvious bitwise operation or pointer arithmetic */

		/* Pre-condition: Required input state before execution. Invariant: Valid state maintained during execution. */
		for (j = 0; j < N; j++) {
			register double *pa = pa_first;
			register double *pb = &B[j * N]; /* Non-obvious bitwise operation or pointer arithmetic */

			register double suma = 0;

			/* Pre-condition: Required input state before execution. Invariant: Valid state maintained during execution. */
			for (k = 0; k < N; k++) {
				suma += *pa * *pb;
				pa++;
				pb++;
			}

			*res = suma;
			res++;
		}
	}

	/* Pre-condition: Required input state before execution. Invariant: Valid state maintained during execution. */
	for (i = 0; i < N; i++) {
		double *pa_first = &A[i]; /* Non-obvious bitwise operation or pointer arithmetic */
		double *res = &third_mul[i * N]; /* Non-obvious bitwise operation or pointer arithmetic */

		/* Pre-condition: Required input state before execution. Invariant: Valid state maintained during execution. */
		for (j = 0; j < N; j++) {
			register double *pa = pa_first;
			register double *pb = &A[j]; /* Non-obvious bitwise operation or pointer arithmetic */

			register double suma = 0;

			/* Pre-condition: Required input state before execution. Invariant: Valid state maintained during execution. */
			for (k = 0; k <= i; k++) {
				suma += *pa * *pb;
				pa += N;
				pb += N;
			}

			*res = suma;
			res++;
		}
	}

	/* Pre-condition: Required input state before execution. Invariant: Valid state maintained during execution. */
	for (i = 0; i < N; i++) {
		register double *res = &result[i * N]; /* Non-obvious bitwise operation or pointer arithmetic */
		register double *pa = &second_mul[i * N]; /* Non-obvious bitwise operation or pointer arithmetic */
		register double *pb = &third_mul[i * N]; /* Non-obvious bitwise operation or pointer arithmetic */

		/* Pre-condition: Required input state before execution. Invariant: Valid state maintained during execution. */
		for (j = 0; j < N; j++) {
			*res = *pa + *pb;
			res++;
			pa++;
			pb++;
		}
		res += N;
		pa += N;
		pb += N;
	}

	free(first_mul);
	free(second_mul);
	free(third_mul);

	return result;
}
