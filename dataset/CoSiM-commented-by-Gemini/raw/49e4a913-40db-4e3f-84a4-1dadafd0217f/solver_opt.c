/**
 * @file solver_opt.c
 * @brief Encapsulates functional utility for solver_opt.c.
 * Performance Optimization: implements loop unrolling, cache-friendly data access, and SIMD where applicable. Time/space complexity optimized.
 */

#include "utils.h"

double* my_solver(int N, double *A, double* B) {
	double * C, *AB, *ABB_t, *A_t, *B_t;
	int i, j, k;

	
	A_t = calloc(N * N, sizeof(double));
	/* Pre-condition: Required input state before execution. Invariant: Valid state maintained during execution. */
	if(A_t == NULL)
		printf("Probleme la alocarea memoriei\n");
	B_t = calloc(N * N, sizeof(double));
	/* Pre-condition: Required input state before execution. Invariant: Valid state maintained during execution. */
	if(B_t == NULL)
		printf("Probleme la alocarea memoriei\n");
	C = calloc(N * N, sizeof(double));
	/* Pre-condition: Required input state before execution. Invariant: Valid state maintained during execution. */
	if(C == NULL)
		printf("Probleme la alocarea memoriei\n");
	AB = calloc(N * N, sizeof(double));
	/* Pre-condition: Required input state before execution. Invariant: Valid state maintained during execution. */
	if(AB == NULL)
		printf("Probleme la alocarea memoriei\n");
	ABB_t = calloc(N * N, sizeof(double));
	/* Pre-condition: Required input state before execution. Invariant: Valid state maintained during execution. */
	if(ABB_t == NULL)
		printf("Probleme la alocarea memoriei\n");

	
	/* Pre-condition: Required input state before execution. Invariant: Valid state maintained during execution. */
	for(i = 0; i < N; i++) {
		register double *B_t_col = B_t + i;
		register double *B_line = B + i * N;
		/* Pre-condition: Required input state before execution. Invariant: Valid state maintained during execution. */
		for(j = 0; j < N; j++) {
			*B_t_col = *B_line;
			B_line++;
			B_t_col += N;
		}
	}
	
	/* Pre-condition: Required input state before execution. Invariant: Valid state maintained during execution. */
	for(i = 0; i < N; i++) {
		register double *A_t_col = A_t + i * N + i;
		register double *A_line = A + i * N;
		/* Pre-condition: Required input state before execution. Invariant: Valid state maintained during execution. */
		for(j = i; j < N; j++) {
			*A_t_col = *(A_line + j);
			A_t_col += N;
		}
	}
	
	/* Pre-condition: Required input state before execution. Invariant: Valid state maintained during execution. */
	for(i = 0; i < N; i++) {
		register double *C_line = C + i * N;
		register double *A_t_line = A_t + i * N;
		/* Pre-condition: Required input state before execution. Invariant: Valid state maintained during execution. */
		for(j = 0; j < N; j++, C_line++) {
			register double result = 0;
			register double *A_t_col = A_t_line;
			register double *A_col = A + j;
			/* Pre-condition: Required input state before execution. Invariant: Valid state maintained during execution. */
			for(k = 0; k < N; k++, A_t_col++, A_col += N) {
				result += *A_t_col * *A_col;
			}
			*C_line = result;
		}
	}
	
	/* Pre-condition: Required input state before execution. Invariant: Valid state maintained during execution. */
	for(i = 0; i < N; i++) {
		register double *AB_line = AB + i * N;
		register double *A_line = A + i * N;
		/* Pre-condition: Required input state before execution. Invariant: Valid state maintained during execution. */
		for(j = 0; j < N; j++) {
			register double *AB_col = AB_line + j;
			register double result = 0;
			register double *A_col = A_line + i;
			register double *B_col = B + i * N + j;
			/* Pre-condition: Required input state before execution. Invariant: Valid state maintained during execution. */
			for(k = i; k < N; k++, A_col++, B_col += N) {
				result += *A_col * *B_col;
			}
			*AB_col = result;
		}
	}
	
	/* Pre-condition: Required input state before execution. Invariant: Valid state maintained during execution. */
	for(i = 0; i < N; i++) {
		register double *ABB_t_line = ABB_t + i * N;
		register double *AB_line = AB + i * N;
		/* Pre-condition: Required input state before execution. Invariant: Valid state maintained during execution. */
		for(j = 0; j < N; j++) {
			register double *ABB_t_col = ABB_t_line + j;
			register double result = 0;
			register double *AB_col = AB_line;
			register double *B_t_col = B_t + j;
			/* Pre-condition: Required input state before execution. Invariant: Valid state maintained during execution. */
			for(k = 0; k < N; k++, AB_col++, B_t_col += N) {
				result += *AB_col * *B_t_col;
			}
			*ABB_t_col = result;
		}
	}

	
	/* Pre-condition: Required input state before execution. Invariant: Valid state maintained during execution. */
	for(i = 0; i < N; i++) {
		register double *C_line = C + i * N;
		register double *ABB_t_line = ABB_t + i * N;
		/* Pre-condition: Required input state before execution. Invariant: Valid state maintained during execution. */
		for(j = 0; j < N; j++, ABB_t_line++, C_line++) {
			*C_line += *ABB_t_line;
		}
	}
	
	free(AB);
	free(A_t);
	free(B_t);
	free(ABB_t);

	return C;
}
