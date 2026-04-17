/* Module Level: Non-optimized matrix solver. @raw/273e4698-fe2e-4c21-9ab2-9163ae0bc6d5/solver_neopt.c */
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

	
	/* Block Level: Calculate AtA */
	for (i = 0; i < N; i ++) {
		for (j = 0 ;j < N; j ++) {
			for (k = 0; k <= i; k ++) {
				AtA[i * N + j] += A[k * N + j] * A[k * N + i];
			}
		}
	}

	
	/* Block Level: Calculate BBt */
	for (i = 0; i < N; i ++) {
		for (j = 0 ;j < N; j ++) {
			for (k = 0; k < N; k ++) {
				BBt[i * N + j] += B[i * N + k] * B[j * N + k];
			}
		}
	}

	
	/* Block Level: Calculate ABBt */
	for (i = 0; i < N; i ++) {
		for (j = 0 ;j < N; j ++) {
			for (k = i; k < N; k ++) {
				ABBt[i * N + j] += A[i * N + k] * BBt[k * N + j];
			}
		}
	}

	
	/* Block Level: Final matrix addition */
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