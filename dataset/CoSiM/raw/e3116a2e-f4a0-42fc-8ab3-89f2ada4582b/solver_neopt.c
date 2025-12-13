
#include "utils.h"


double* my_solver(int N, double *A, double* B) {
	printf("NEOPT SOLVER\n");
	
	int i, j, k;
	double *C = calloc(N*N,sizeof(double));
	
	double *AUX = calloc(N*N, sizeof(double));
	
	double res;
	

	
	for(i = 0 ; i < N; i++){
		for(j = 0; j < N; j++){
			res = 0;
			for(k = 0; k < N; k++){
				
				if(i <= k)
					res += A[i*N + k] * B[k*N + j];
			}
			AUX[i*N + j] = res;
		}
	}
	
	for(i = 0 ; i < N; i++){
		for(j = 0; j < N; j++){
			res = 0;
			for(k = 0; k < N; k++){
					res += AUX[i*N + k] * B[j*N + k];
					
					if(k <= j)
						res += A[k*N + i] * A[k*N + j];
			}
			C[i*N + j] = res;
		}
	}

	free(AUX);
	return C;
}
