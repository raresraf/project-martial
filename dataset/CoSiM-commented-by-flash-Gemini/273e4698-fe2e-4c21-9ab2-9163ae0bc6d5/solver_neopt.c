/**
 * @file solver_neopt.c
 * @brief Semantic documentation for solver_neopt.c. 
 *        This is a placeholder. Detailed semantic analysis will be applied later.
 */

#include "utils.h"


double* my_solver(int N, double *A, double* B) {
	printf("NEOPT SOLVER\n");
	double *C, *BBt, *ABBt, *AtA; 
	int i, j, k;
	C    = calloc(N * N, sizeof(double));
	BBt  = calloc(N * N, sizeof(double));
	ABBt = calloc(N * N, sizeof(double));
	AtA  = calloc(N * N, sizeof(double));

	if((C == NULL) || (AtA == NULL) || (ABBt == NULL) || (BBt == NULL)) {
		return NULL;
	}

	
	for (i = 0; i < N; i ++) {
		for (j = 0 ;j < N; j ++) {
			for (k = 0; k <= i; k ++) {
				AtA[i * N + j] += A[k * N + j] * A[k * N + i];
			}
		}
	}

	
	for (i = 0; i < N; i ++) {
		for (j = 0 ;j < N; j ++) {
			for (k = 0; k < N; k ++) {
				BBt[i * N + j] += B[i * N + k] * B[j * N + k];
			}
		}
	}

	
	for (i = 0; i < N; i ++) {
		for (j = 0 ;j < N; j ++) {
			for (k = i; k < N; k ++) {
				ABBt[i * N + j] += A[i * N + k] * BBt[k * N + j];
			}
		}
	}

	
	for (i = 0; i < N; i ++) {
		for (j = 0; j < N; j ++) {
			C[i * N + j] = ABBt[i * N + j] + AtA[i * N + j];
		}
	}
	
	free(AtA);
	free(ABBt);
	free(BBt);

	return C;
}

