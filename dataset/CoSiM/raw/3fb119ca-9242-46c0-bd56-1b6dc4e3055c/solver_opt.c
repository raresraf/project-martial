
#include "utils.h"
#define min(x, y) (((x) < (y)) ? (x) : (y))

double* my_solver(int N, double *A, double* B) {
	register double * AB, *ABBt, * AtA, * result, *Bt, *At;
	register int i, j, k;
	AB = malloc(N * N * sizeof(double));
	ABBt = malloc(N * N * sizeof(double));
	AtA = malloc(N * N * sizeof(double));
	result = malloc(N * N * sizeof(double));
	Bt = malloc(N * N * sizeof(double));
	At = malloc(N * N * sizeof(double));

	for(i = 0; i < N; i++) {
		register double *orig_pA = A + i * N;;
		for(j = 0; j < N; j++) {
			register double *pA = orig_pA + i;
   			register double *pB = B + i * N + j;
   			register double current_result = 0;
   			for(k = i; k < N; k++) {
   				current_result += *pA * *pB;
   				pA += 1;
   				pB += N;
   			}
			AB[i * N + j] = current_result;
		}
	}
	for (i = 0; i < N; i++) {
		register double *pA = A + i * N;  
		register double *pAt = At + i;
		register double *pB = B + i * N;   
		register double *pBt= Bt + i;   
		for (j = 0; j < N; j++) {
			*pAt = *pA;
			*pBt = *pB;
			pAt += N;
			pBt += N;
			pA += 1;
			pB += 1;
		}
	}
	for(i = 0; i < N; i++) {
		register double *orig_pAB = AB + i * N;;
		for(j = 0; j < N; j++) {
			register double *pAB = orig_pAB;
   			register double *pBt = Bt + j;
   			register double current_result = 0;
   			for(k = 0; k < N; k++) {
   				current_result += *pAB * *pBt;
   				pAB += 1;
   				pBt += N;
   			}
			ABBt[i * N + j] = current_result;
		}
	}
	for(i = 0; i < N; i++) {
		register double *orig_pAt = At + i * N;;
		for(j = 0; j < N; j++) {
			register double *pAt = orig_pAt;
   			register double *pA = A + j;
   			register double current_result = 0;
   			for(k = 0; k <= min(i, j); k++) {
   				current_result += *pAt * *pA;
   				pAt += 1;
   				pA += N;
   			}
			AtA[i * N + j] = current_result;
		}
	}

	for (i = 0; i < N; ++i) {
		register double *pABBt = ABBt + N * i;
		register double *pAtA = AtA + N * i;
		register double *p_result = result + N * i;
		for (j = 0; j < N; ++j) {
			*p_result = *pABBt + *pAtA;
			p_result += 1;
			pABBt += 1;
			pAtA += 1;
		}
	}

	free(AB);
	free(ABBt);
	free(AtA);
	free(At);
	free(Bt);

	return result;


}
