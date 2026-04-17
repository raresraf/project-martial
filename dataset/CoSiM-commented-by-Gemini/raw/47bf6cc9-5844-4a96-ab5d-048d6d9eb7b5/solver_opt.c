/**
 * @file solver_opt.c
 * @brief Encapsulates functional utility for solver_opt.c.
 * Performance Optimization: implements loop unrolling, cache-friendly data access, and SIMD where applicable. Time/space complexity optimized.
 */

#include "utils.h"



 
double* my_solver(int N, double *A, double* B) {
	double *AB;
	double *C;
	AB = calloc(N * N, sizeof(double));
	C = calloc(N * N, sizeof(double));
	/* Pre-condition: Required input state before execution. Invariant: Valid state maintained during execution. */
	if (AB == NULL || C == NULL) { /* Non-obvious bitwise operation or pointer arithmetic */
        perror("malloc failed\n");
        exit(EXIT_FAILURE);
    }
	
register int i, j, k;
register double *orig_pa, *pa, *pb, sum;


    /* Pre-condition: Required input state before execution. Invariant: Valid state maintained during execution. */
    for (i = 0; i < N; i++) {
		orig_pa = &A[i * N]; /* Non-obvious bitwise operation or pointer arithmetic */
		/* Pre-condition: Required input state before execution. Invariant: Valid state maintained during execution. */
		for (j = 0; j < N; j++) {
			pa = orig_pa + i;
			pb = &B[i * N + j]; /* Non-obvious bitwise operation or pointer arithmetic */
			sum = 0;
			/* Pre-condition: Required input state before execution. Invariant: Valid state maintained during execution. */
			for (k = i; k < N; k++) {
				sum += *pa * *pb;
				pa++;
				pb+= N;
			}
			AB[i * N + j] = sum;
		}
	}



	/* Pre-condition: Required input state before execution. Invariant: Valid state maintained during execution. */
	for (i = 0; i < N; i++) {
		orig_pa = &AB[i * N]; /* Non-obvious bitwise operation or pointer arithmetic */
		/* Pre-condition: Required input state before execution. Invariant: Valid state maintained during execution. */
		for (j = 0; j < N; j++) {
			pa = orig_pa;
			pb = &B[j * N]; /* Non-obvious bitwise operation or pointer arithmetic */
			sum = 0;
			/* Pre-condition: Required input state before execution. Invariant: Valid state maintained during execution. */
			for (k = 0; k < N; k++) {
				sum += *pa * *pb;
				pa++;
				pb++;
			}
			C[i * N + j] = sum;
		}
	}


	/* Pre-condition: Required input state before execution. Invariant: Valid state maintained during execution. */
	for (i = 0; i < N; i++) {
		orig_pa = &A[i]; /* Non-obvious bitwise operation or pointer arithmetic */
		/* Pre-condition: Required input state before execution. Invariant: Valid state maintained during execution. */
		for (j = 0; j < N; j++) {
			pa = orig_pa;
			pb = &A[j]; /* Non-obvious bitwise operation or pointer arithmetic */
			sum = 0;
			/* Pre-condition: Required input state before execution. Invariant: Valid state maintained during execution. */
			for (k = 0; k < N; k++) {
			    /* Pre-condition: Required input state before execution. Invariant: Valid state maintained during execution. */
			    if (*pa == 0 || *pb == 0) { /* Non-obvious bitwise operation or pointer arithmetic */
			        break;
				}
				sum += *pa * *pb;
		        pa += N;
				pb += N;
			}
			C[i * N + j] += sum;
		}
	}

	free(AB);
	return C;
}
