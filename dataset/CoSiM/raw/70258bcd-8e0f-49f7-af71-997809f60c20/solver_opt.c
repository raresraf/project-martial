
#include "utils.h"


double* my_solver(int N, double *A, double* B) {
	register int i, j, k;
	printf("OPT SOLVER\n");

	
	double *AB = (double*) calloc(N * N, sizeof(double));
	if (AB == NULL) {
		printf("Error calloc");
		return NULL;
	}
	double *ABBt = (double*) calloc(N * N, sizeof(double));
	if (ABBt == NULL) {
		printf("Error calloc");
		return NULL;
	}

	double *AtA = (double*) calloc(N * N, sizeof(double));
	if (AtA == NULL) {
		printf("Error calloc");
		return NULL;
	}

	double *C = (double*) calloc(N * N, sizeof(double));
	if (C == NULL) {
		printf("Error calloc");
		return NULL;
	}


	
	for (i = 0; i < N; ++i) {
		
		register double *orig_pA = &A[i * N + i];

		for (j = 0; j < N; ++j) {
			register double sum = 0.0;
			register double *pA = orig_pA;
			
			register double *pB = &B[i * N + j];

			for (k = i; k < N; ++k) {
				sum += *pA * *pB;
				pA++;
				pB += N;
			}

			AB[i * N + j] = sum;
		}
	}

	
	for (i = 0; i < N; ++i) {
		
		register double *orig_pAt = &A[i];

		for (j = 0; j < N; ++j) {
			register double sum = 0.0;
			register double *pAt = orig_pAt;
			
			register double *pA = &A[j];

			for (k = 0; k <= i ; ++k) {
				sum += *pAt * *pA;
				pAt += N;
				pA += N;
			}

			AtA[j * N + i] = sum;
		}
	}

	
	for (i = 0; i < N; ++i) {
		
		register double *orig_pAB = &AB[i * N];

		for (j = 0; j < N; ++j) {
			register double sum = 0.0;
			register double *pAB = orig_pAB;
			
			register double *pB = &B[j * N];

			for (k = 0; k < N; ++k) {
				sum += *pAB * *pB;
				pAB++;
				pB++;
			}

			C[i * N + j] = sum + AtA[i * N + j];
		}
	}

	
	free(AB);
	free(ABBt);
	free(AtA);

	return C;	
}
