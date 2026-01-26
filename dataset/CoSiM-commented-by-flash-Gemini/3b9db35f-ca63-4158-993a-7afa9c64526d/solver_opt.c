
#include "utils.h"



double* my_solver(int N, double *A, double* B) {
	
	double *At, *Bt, *sum, *sum2, *sum3, *res;
	int i = 0, j = 0, k = 0;
	At = (double *)calloc(N * N, sizeof(double));
	Bt = (double *)calloc(N * N, sizeof(double));
	sum = (double *)calloc(N * N, sizeof(double));
	sum2 = (double *)calloc(N * N, sizeof(double));
	sum3 = (double *)calloc(N * N, sizeof(double));
	res = (double *)calloc(N * N, sizeof(double));
	
	for (i = 0; i < N; i++){
		for (j = 0; j < N; j++){
			register int idx = i * N + j;
			
			if (i >= j){
				At[idx] = A[j * N + i];
			}
			Bt[idx] = B[j * N + i];
		}
	}

	

	for (i = 0; i < N; i++){
		register double *orig_pa = &A[i * N];
		for (k = i; k < N; k++){
			for (j = 0; j < N; j++){
				register int idx = i * N + j;
				register double *orig_pb = &B[k * N];
				sum[idx] += orig_pa[k] * orig_pb[j];
			
			}

		}
	}
	

	for (i = 0; i < N; i++){
		register double *orig_pa = &sum[i * N];
		for (k = 0; k < N; k++){
			for (j = 0; j < N; j++){
				register int idx = i * N + j;
				register double *orig_pb = &Bt[k * N];
				sum2[idx] += orig_pa[k] * orig_pb[j];
			}
			
		}
	}
	

	for (i = 0; i < N; i++){
		register double *orig_pa = &At[i * N];
		for (k = 0; k < N; k++){			
			for (j = k; j < N; j++){
				register int idx = i * N + j;
				register double *orig_pb = &A[k * N];
				sum3[idx] += orig_pa[k] * orig_pb[j];
				
			}
		}
	}
	for (i = 0; i < N; i++){
		for (j = 0; j < N; j++){
			register int idx = i * N + j;
			res[idx] = sum2[idx] + sum3[idx];
			
		}

	}
	
	free(At);
	free(Bt);
	free(sum);
	free(sum2);
	free(sum3);
	return res;
}
