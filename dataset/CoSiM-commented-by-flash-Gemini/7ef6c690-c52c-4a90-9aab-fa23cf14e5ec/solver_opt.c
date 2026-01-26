
#include "utils.h"
#include "string.h"


double* my_solver(int N, double *A, double* B) {
	printf("OPT SOLVER\n");
	double *C = malloc(N * N * sizeof(*C));
	double *aux = calloc(N * N, sizeof(*aux));
	double *At = malloc(N * N * sizeof(*At));
	double *pA, *pB, *pC, *paux; 
	int i, j, k;

	
	for (i = 0; i < N; ++i) {
		pB = B + i * N;
		for (k = i; k < N; ++k) {
			register double a = A[i * N + k];
			paux = aux + i * N;
			for (j = 0; j < N; ++j) {
				*paux += a * *pB;
				++paux;
				++pB;
			}
		}
	}

	
	pC = C;
	for (i = 0; i < N; ++i) {
		pB = B;
		for (j = 0; j < N; ++j) {
			register double s = 0;
			paux = aux + i * N;
			for (k = 0; k < N; ++k) {
				s += *paux * *pB;
				pB++;
				paux++;
			}
			*pC = s;
			++pC;
		}
	}

	
	pA = A;
	for (i = 0; i < N; ++i) {
		double *pAt = At + i;
		for (j = 0; j < N; ++j) {
			*pAt = *pA;
			++pA;
			pAt += N;
		}
	}


	
	paux = aux;
	for (i = 0; i < N; ++i) {
		for (j = 0; j < N; ++j) {
			register double s = 0;
			double *p1 = At + i * N;
			double *p2 = At + j * N;
			for (k = 0; k < i + 1; ++k) {
				s += *p1 * *p2;
				++p1;
				++p2;
			}
			*paux = s;
			++paux;
		}
	}

	
	pC = C;
	paux = aux;
	for (i = 0; i < N; ++i) {
		for (j = 0; j < N; ++j) {
			*pC += *paux;
			++pC;
			++paux;
		}
	}

	free(aux);
	free(At);
	return C;
}
