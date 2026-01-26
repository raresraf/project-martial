
#include "utils.h"


double* my_solver(int N, double *A, double* B) {
	printf("OPT SOLVER\n");

	int i, j, k;
	int idx;
	register double sum;

	double *At = (double *) calloc(N * N, sizeof(double));
	if (At == NULL) 
		return NULL;

	double *Bt = (double *) calloc(N * N, sizeof(double));
	if (Bt == NULL) 
		return NULL;

	double *AxB = (double *) calloc(N * N, sizeof(double));
	if (AxB == NULL) 
		return NULL;

	double *AxBxBt = (double *) calloc(N * N, sizeof(double));
	if (AxBxBt == NULL) 
		return NULL;

	double *result = (double *) calloc(N * N, sizeof(double));
	if (result == NULL) 
		return NULL;
	
	
	for (i = 0; i < N; ++i) {
		register double * a = &A[i * N];
		register double * b = &B[i * N];
		for (j = 0; j < N; ++j) {
			idx = j * N + i;
		
			At[idx] = *a;
			Bt[idx] = *b;

			a++;
			b++;
		}
	}

	
	for (i = 0; i < N; ++i) {
		register double  *a = &A[i * N + i];
		for (j = 0; j < N; ++j) {
			sum = 0.0;
			register double *a2 = a;
			register double *b = &B[i * N + j];
			
			for (k = i; k < N; ++k) {
				sum += *a2 * *b;
				a2++;
				b += N;
			}

			AxB[i * N + j] = sum;
		}
	}
	
	
	for (i = 0; i < N; ++i) {
		register double  *a = &AxB[i * N];
		for (j = 0; j < N; ++j) {
			sum = 0.0;

			register double *a2 = a;
			register double *bt = &Bt[j];
			
			for (k = 0; k < N; ++k) {
				sum += *a2 * *bt;
				a2++;
				bt += N;
			}

			AxBxBt[i * N + j] = sum;
		}
	}

	
	for (i = 0; i < N; ++i) {
		register double  *at = &At[i * N];
		for (j = 0; j < N; ++j) {
			sum = 0.0;
			register double *at2 = at;
			register double *a = &A[j];
			
			for (k = 0; k < i + 1; ++k) {
				sum += *at2 * *a;
				at2++;
				a += N;
			}

			result[i * N + j] = sum;
		}
	}

	for (i = 0; i < N; ++i) {
		register double *a = &AxBxBt[i * N];
		for (j = 0; j < N; ++j) {
			result[i * N + j] += *a;
			a++;
		}
	}
	return result;
}
