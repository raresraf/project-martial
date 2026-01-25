
#include "utils.h"


double* my_solver(int N, double *A, double* B) {
	double *C, *A_tA, *AB;
	register int i, j, k;
	
	C = calloc(N * N, sizeof(double));
	if (C == NULL)
		return NULL;
	A_tA = calloc(N * N, sizeof(double));
	if (A_tA == NULL)
		return NULL;
	AB = calloc(N * N, sizeof(double));
	if (AB == NULL)
		return NULL;
	
	for (k = 0; k < N; k++) {
		register double *orign_At = &A[k * N];
		for (i = k; i < N; i++) {
			register double *At = orign_At + i;
			register double *pA = orign_At + k;
			register double *pAt_A = &A_tA[i * N + k]; 
			for (j = k; j < N; j++) {
				*pAt_A += *At * *pA;
				pA++;
				pAt_A++;
			}
		}
	}
	
	for (i = 0; i < N; i++) {
		register double *orign_A = &A[i * N];
		register double *orign_AB = &AB[i * N];
		for (k = i; k < N; k++) {
			register double *pA2 = orign_A + k;
			register double *pB = B + k * N;
			register double *pAB = orign_AB;
			for (j = 0; j < N; j++) {
				*pAB += *pA2 * *pB;
				pB++;
				pAB++;
			}
		}
	}
	
	
	for (i = 0; i < N; i++) {
		register double *orign_C = &C[i * N];
		register double *orign_AB = &AB[i * N];
		for (j = 0; j < N; j++) {
			register double *pC = orign_C + j;
			register double *pAB = orign_AB;
			register double *pB = &B[j * N];
			for (k = 0; k < N; k++) {
				*pC += *pAB * *pB;
				pAB++;
				pB++;
			}
		}
	}
	
	for (i = 0; i < N; i++) {
		register double *pC = &C[i * N];
		register double *pA_tA = &A_tA[i * N];
		for (j = 0;  j < N; j++) {
			*pC += *pA_tA;
			pC++;
			pA_tA++;
		}
	}
	free(A_tA);
	free(AB);
	return C;
}
