
#include "utils.h"


double* my_solver(int N, double *A, double* B) {
	printf("OPT SOLVER\n");
	register int i, j, k;

	
	double* C = calloc(N * N, sizeof(double));
	double* AB = calloc(N * N, sizeof(double)); 
	double* prod1 = calloc(N * N, sizeof(double)); 
	double* prod2 = calloc(N * N, sizeof(double)); 
	if (C == NULL || AB == NULL || prod1 == NULL || prod2 == NULL) {
        perror("calloc failed\n");
        exit(EXIT_FAILURE);
    }


	for (i = 0; i < N; i++) {
		
		
		register double* lineA = A + i * N + i;
		register double* pAB = AB + i * N;
		for (j = 0; j < N; j++) {
			register double sum = 0.0;
			register double* elemA = lineA;
			register double* columnB = B + i * N + j;
			for (k = i; k < N; k++) {
				
				sum += *elemA * *columnB;
				elemA++;
				columnB += N;
			}
			*pAB = sum;
			pAB++;
		}
	}

	for (i = 0; i < N; i++) {
		register double* lineAB = AB + i * N;
		register double* pProd1 = prod1 + i * N;
		for (j = 0; j < N; j++) {
			register double sum = 0.0;
			register double* elemAB = lineAB;
			register double* elemBt = B + j * N;
			for (k = 0; k < N; k++) {
				
				sum += *elemAB * *elemBt;
				elemAB++;
				elemBt++;
			}
			*pProd1 = sum;
			pProd1++;
		}
	}

	for (i = 0; i < N; i++) {
		register double* columnAt = A + i;
		register double* pProd2 = prod2 + i * N;
		for (j = 0; j < N; j++) {
			register double sum = 0.0;
			register double* elemAt = columnAt;
			register double* elemA = A + j;
			for (k = 0; k < N; k++) {
				
				sum += *elemAt * *elemA;
				elemA += N;
				elemAt += N;
			}
			*pProd2 = sum;
			pProd2++;
		}
	}

	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			C[i * N + j] = prod1[i * N + j] + prod2[i * N + j];
		}
	}

	free(AB);
	free(prod1);
	free(prod2);
	return C;	
}