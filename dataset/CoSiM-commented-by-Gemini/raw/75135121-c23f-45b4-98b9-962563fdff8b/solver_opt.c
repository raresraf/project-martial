
#include "utils.h"



double* compute_transpose(int N, double* M)
{
	double* res = (double*)malloc(N * N * sizeof(double));
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			res[j * N + i] = M[i * N + j];
		}
	}
	return res;
}

double* my_solver(int N, double *A, double* B) {
	printf("OPT SOLVER\n");

	double* res_AxB = (double*)malloc(N * N * sizeof(double));
	double* res_ABBt = (double*)malloc(N * N * sizeof(double));
	double* res_AtA = (double*)malloc(N * N * sizeof(double));
	double* res = (double*)malloc(N * N * sizeof(double));
	double* B_t = compute_transpose(N, B);
	double* A_t = compute_transpose(N, A);
	register int i, j, k;
	double* pa = 0;
	double* pb = 0;
	
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			pa = &A[i * N + i];
			pb = &B[i * N + j];
			register double suma = 0;
			for (k = i; k < N; k++) {
				suma += *pa * *pb;
				pa++;
				pb += N;
			}
			
			res_AxB[i * N + j] = suma;
		}
	}

	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			pa = &res_AxB[i * N];
			pb = &B_t[j];
			register double suma = 0;
			for (k = 0; k < N; k++) {
				suma += *pa * *pb;
				pa++;
				pb += N;
			}
			
			res_ABBt[i * N + j] = suma;
		}
	}
	
	free(res_AxB);
	free(B_t);
	int P = 0;
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			pa = &A_t[i * N];
			pb = &A[j];
			if (i < j)
				P = i;
			else
				P = j;
			register double suma = 0;
			for (k = 0; k <= P; k++) {
				suma += *pa * *pb;
				pa++;
				pb += N;
			}
			res[i * N + j] = 0;
			res_AtA[i * N + j] = suma;
			res[i * N + j] += res_ABBt[i * N + j] + res_AtA[i * N + j];
		}
	}
	free(A_t);
	
	free(res_ABBt);
	free(res_AtA);
	return res;	
}
