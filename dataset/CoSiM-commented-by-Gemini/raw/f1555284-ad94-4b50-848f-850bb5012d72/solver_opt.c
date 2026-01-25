
#include "utils.h"

double* my_solver(int N, double *A, double *B) {
	
	double *C = (double *)malloc(N * N * sizeof(double));
	double *D = (double *)malloc(N * N * sizeof(double));

	
	for (int i = 0; i < N; i++) {
		double *ptrC = &(C[i * N]);
		for (int j = 0; j < N; j++) {
			register double sum = 0.0;
			double *ptr = &(A[i * N + i]);
			double *ptrB = &(B[i * N + j]);
			for (int k = i; k < N; k++) {
				sum += *ptr * (*ptrB);
				ptr++;
				ptrB += N;
			}
			*ptrC = sum;
			ptrC++;
		}
	}

	
	for (int i = 0; i < N; i++) {
		double *ptrD = &(D[i * N]);
		for (int j = 0; j < N; j++) {
			register double sum2 = 0.0;
			double *ptrC = &(C[i * N]);
			double *ptrB = &(B[j * N]);
			for (int k = 0; k < N; k++) {
				sum2 += *ptrC * *ptrB;
				ptrC++;
				ptrB++;
			}
			*ptrD = sum2;
			ptrD++;
		}
	}

	
	for (int i = 0; i < N; i++) {
		double *ptrC = &(C[i * (N + 1)]);
		for (int j = i; j < N; j++) {
			register double sum3 = 0.0;
			double *ptrA1 = &(A[j]);
			double *ptrA2 = &(A[i]);
			for (int k = 0; k <= i; k++) {
				sum3 += *ptrA1 * (*ptrA2);
				ptrA1 += N;
				ptrA2 += N; 
			}
			*ptrC = sum3;
			ptrC++;
            if (i != j)
                C[j * N + i] = sum3;
		}
	}
	
	int size = N * N;
	double *ptrC = &(C[0]);
	double *ptrD = &(D[0]);

	
	for (int i = 0; i < size; i++) {
		*ptrC += *ptrD;
		ptrC++;
		ptrD++;
	}

	free(D);

	return C;
}