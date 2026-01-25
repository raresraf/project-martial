/**
 * @file solver_opt.c
 * @brief Optimized implementation of a matrix solver.
 *
 * This file contains an optimized version of the `my_solver` function. It
 * employs several optimization techniques, such as using `register` variables
 * and pointer arithmetic, to improve performance over the naive implementation.
 * These techniques aim to reduce memory access latency by encouraging the
 * compiler to keep frequently accessed variables in CPU registers.
 */
#include "utils.h"


/**
 * @brief Solves a matrix equation using an optimized approach.
 * @param N The dimension of the matrices.
 * @param A A pointer to the first input matrix.
 * @param B A pointer to the second input matrix.
 * @return A pointer to the resulting matrix.
 *
 * This function computes the matrix expression `A * B * B' + A' * A`, where `A'`
 * and `B'` are the transposes of matrices `A` and `B`, respectively. The
 * implementation is optimized with register variables and pointer arithmetic
 * to improve cache locality and reduce memory access overhead.
 */
double* my_solver(int N, double *A, double* B) {
	register int i, j, k;

	printf("OPT SOLVER\n");

	double *at = calloc(N * N, sizeof(double));
	if (at == NULL)
		exit(EXIT_FAILURE);

	double *bt = calloc(N * N, sizeof(double));
	if (bt == NULL)
		exit(EXIT_FAILURE);

	double *res1 = calloc(N * N, sizeof(double));
	if (res1 == NULL)
		exit(EXIT_FAILURE);

	double *res2 = calloc(N * N, sizeof(double));
	if (res2 == NULL)
		exit(EXIT_FAILURE);

	double *res3 = calloc(N * N, sizeof(double));
    if (res3 == NULL)
            exit(EXIT_FAILURE);

	double *res = calloc(N * N, sizeof(double));
	if (res == NULL)
        exit(EXIT_FAILURE);


	
	// Computes the transpose of matrices A and B using optimized pointer arithmetic.
	for (i = 0; i < N; i++) {

		
		register double *p_at = at + i;
		register double *p_bt = bt + i;
		register double *ptA = A + i * N;
		register double *ptB = B + i * N;

		for (j = 0; j < N; j++) {
			*p_at = *ptA;
			*p_bt = *ptB;
			p_at += N;
			p_bt += N;
			ptA++;
			ptB++;
		}
	}

	
	// Computes the matrix multiplication res1 = A * B with loop optimization.
	for (i = 0; i < N; i++) {
		register double *pt_to_A = &A[i * N + i];
		register double *p_res1 = &res1[i * N];
		for (j = 0; j < N; j++) {
			
			register double sum = 0.0;
			
			register double *pa = pt_to_A;
			register double *pb = &B[i*N + j];
			for (k = i; k < N; k++) {
				sum += *pa * *pb;
				pa++;
				pb += N;
			}
			*(p_res1 + j) = sum;
		}
	}

	
	// Computes the matrix multiplication res2 = res1 * B' with loop optimization.
	for (i = 0; i < N; i++) {
		register double *pt_to_res1 = &res1[i * N];
		for (j = 0; j < N; j++) {
			
			register double sum = 0.0;
			
			register double *pres1 = pt_to_res1;
			register double *pbt = &bt[j];
			for (k = 0; k < N; k++) {
				sum += *pres1 * *pbt;
				pres1++;
				pbt += N;
			}
			res2[i * N + j]= sum;
		}
	}

	
	// Computes the matrix multiplication res3 = A' * A with loop optimization.
	for (i = 0; i < N; i++) {
		register double *pt_to_at = &at[i * N ];
		for (j = 0; j < N; j++) {
			
			register double sum = 0.0;
			
			register double *pat = pt_to_at;
			register double *pA = &A[j];
			for (k = 0; k <= j; k++) {
				sum += *pat * *pA;
				pat++;
				pA += N;
			}
			res3[i * N + j] = sum ;
		}
	}

	
	// Computes the final result res = res2 + res3.
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			res[i * N + j] = res2[i * N + j] + res3[i * N + j];
		}
	}

	free(at);
	free(bt);
	free(res1);
	free(res2);
	free(res3);
	return res;
}
