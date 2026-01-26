
#include "utils.h"


double* multiply(int N, double *A, double *B, int tr){
	int i = 0, j = 0, k = 0;
	double *mul;
	mul = (double *)calloc(N * N, sizeof(double));
	if (!mul) return NULL;

	for (i = 0; i < N; i++){
		for (j = 0; j < N; j++){
			for (k = 0; k < N; k++){
				
				if (tr == 1){
					
					if (i <= k)
						mul[i * N + j] += A[i * N + k] * B[k * N + j];
					else continue;
				}
				
				else if (tr == 2){
					if (k <= j)
						mul[i * N + j] += A[i * N + k] * B[k * N + j];
					else continue;
				}
				
				else mul[i * N + j] += A[i * N + k] * B[k * N + j];
			}
		}
	}

	return mul;
}
double* my_solver(int N, double *A, double* B) {
	
	double *At, *Bt, *sum, *sum2, *sum3, *c;
	int i = 0, j = 0;
	At = (double *)calloc(N * N, sizeof(double));
	Bt = (double *)calloc(N * N, sizeof(double));
	c = (double *)calloc(N * N, sizeof(double));
	
	for (i = 0; i < N; i++){
		for (j = 0; j < N; j++){
			
			if (i >= j){
				At[i * N + j] = A[j * N + i];
			}
		}
	}
	
	for (i = 0; i < N; i++){
		for (j = 0; j < N; j++){
			Bt[i * N + j] = B[j * N + i];
		}
	}
	
	sum = multiply(N, A, B, 1);
	
	sum2 = multiply(N, sum, Bt, 0);
	
	sum3 = multiply(N, At, A, 2);
	
	for (i = 0; i < N; i++){
		for (j = 0; j < N; j++){
			c[i * N + j] = sum2[i * N + j] + sum3[i * N + j];	
		}
	}
	free(At);
	free(Bt);
	free(sum);
	free(sum2);
	free(sum3);

	return c;
}
