
#include "utils.h"


double* my_solver(int N, double *A, double* B) {
	printf("OPT SOLVER\n");
	
	

	int i, j, k;
	double *A_transpose = (double *) calloc(N * N, sizeof(double));
	double *B_transpose = (double *) calloc(N * N, sizeof(double));
	double *rezPartial1 = (double *) calloc(N * N, sizeof(double));
	double *rez = (double *) calloc(N * N, sizeof(double));
	
	
	for (i = 0; i < N; i++) {
		register double *ptr = &A[i * N + i];
		for (j = i; j < N; j++) {
			register int idx = j * N + i;
			A_transpose[idx] = *ptr;
			ptr++;
		}
		register double *ptr1 = &B[i * N];
		for (j = 0; j < N; j++) {
			register int idx = j * N + i;
			B_transpose[idx] = *ptr1;
			ptr1++;
		}
		
	}

	
	
	for (i = 0; i < N; i++) {
		register int idx1 = i * N;
		register double *ptrA = &A[idx1];
		register double *ptrRez1 = &rezPartial1[idx1];
		register double *ptrAt = &A_transpose[idx1];
		register double *ptrRez2 = &rez[idx1];

		for (k = 0; k < N; k++) {
			register int idx2 = k * N;
			register double temp = *(ptrA + k);
			register double temp1 = *(ptrAt + k);
			register double *ptrB = &B[idx2];
			register double *ptrA2 = &A[idx2];
			register double *ptrRez11 = ptrRez1;
			register double *ptrRez22 = ptrRez2;

			for (j = 0; j < N; j++) {
				*(ptrRez11) += temp * (*ptrB);
				*(ptrRez22) += temp1 * (*ptrA2);
				
				ptrRez11++;
				ptrRez22++;
				ptrB++;
				ptrA2++;
			}
		}
	}
	free(A_transpose);

	
	for (i = 0; i < N; i++) {
		register int idx1 = i * N;
		register double *ptr1 = &rezPartial1[idx1];
		register double *ptrRez = &rez[idx1];
		
		for (k = 0; k < N; k++) {
			register int idx2 = k * N;
			register double temp = *(ptr1 + k);
			register double *ptrBt = &B_transpose[idx2];
			register double *ptrRez1 = ptrRez;

			for (j = 0; j < N; j++) {
				(*ptrRez1) += temp * (*ptrBt);
				ptrRez1++;
				ptrBt++;
			}
		}
	}
	
	free(rezPartial1);
	free(B_transpose);
	return rez;	
}
