
#include "utils.h"
#define UPPER 1
#define LOWER -1
#define NORMAL 0


void transpose(int N, double **C, double *M) {
	register int i, j;
	for(i = 0; i < N; i ++) {
		register double *ptrC = &(*C)[i * N ];
		register double *ptrM = &M[i];
		for(j = 0; j < N; j++) {
			*ptrC = *ptrM;
			ptrM += N;
			ptrC++;
		}
	}
}

void multiply_normal(int N, double **C, double *A, double *B, int typeA, int typeB) {
	register double *ptrC = *C;
	register int i, j, k;
	for(i = 0; i < N; i++) {
		register double *pa = &A[i * N];
		for(j = 0; j < N; j++) {
			register double suma = 0.0;
			register double *ptrA = pa;
			register double *ptrB = &B[j];
			for(k = 0; k < N; k++) {
				suma += *ptrA * *ptrB;
				ptrA++;	
				ptrB += N;	
			}
			*ptrC = suma;
			ptrC++;
		}
	}
}

void multiply_upper(int N, double **C, double *A, double *B, int typeA, int typeB) {
	register double *ptrC = *C;
	register int i, j, k;
	for(i = 0; i < N; i++) {
		register double *pa = &A[i * N];
		for(j = 0; j < N; j++) {
			register double suma = 0.0;
			register double *ptrA = pa;
			register double *ptrB = &B[j];
			for(k = 0; k < N; k++) {
				if(i <= k)
					suma += *ptrA * *ptrB;
				ptrA++;	
				ptrB += N;	
			}
			*ptrC = suma;
			ptrC++;
		}
	}
}

void multiply_lower_upper(int N, double **C, double *A, double *B, int typeA, int typeB) {
	register double *ptrC = *C;
	register int i, j, k;
	for(i = 0; i < N; i++) {
		register double *pa = &A[i * N];
		for(j = 0; j < N; j++) {
			register double suma = 0.0;
			register double *ptrA = pa;
			register double *ptrB = &B[j];
			for(k = 0; k < N; k++) {
				if(i >= k && k <= j)
					suma += *ptrA * *ptrB;
				ptrA++;	
				ptrB += N;	
			}
			*ptrC = suma;
			ptrC++;
		}
	}
}

void add(int N, double **C, double *A, double *B) {
	register double *ptrC = *C;
	register double *ptrA = &A[0];
	register double *ptrB = &B[0];
	register int i;
	for(i = 0; i < N * N; i++) {
		*ptrC = *ptrA + *ptrB;
		ptrA++;
		ptrB++;
		ptrC++;
	}
}

double* my_solver(int N, double *A, double* B) {
	
	double *AB = calloc(N* N,sizeof(double));
	multiply_upper(N, &AB, A, B, UPPER, NORMAL);

	double *Btrans = malloc((N * N) * sizeof(double));
	transpose(N, &Btrans, B);

	double *ABBt = calloc(N * N,sizeof(double));
	multiply_normal(N, &ABBt, AB, Btrans, NORMAL, NORMAL);

	
	double *Atrans = malloc((N * N) * sizeof(double));
	transpose(N, &Atrans, A);
	
	double *AtA = calloc(N* N,sizeof(double));
	multiply_lower_upper(N, &AtA, Atrans, A, LOWER, UPPER);

	double *result = malloc((N * N) * sizeof(double));
	add(N, &result, ABBt, AtA);

	free(AB);
	free(Btrans);
	free(ABBt);
	free(Atrans);
	free(AtA);

	return result;
}
