
#include "utils.h"


double* my_solver(int N, double *A, double* B) {
	printf("NEOPT SOLVER\n");
	double *C1 = calloc(N*N, sizeof(double));
	
	double *C = calloc(N*N, sizeof(double));
	int i,j,k;
	
	
	for(i=0; i<N; i++){
		for(j=0; j<N; j++){
			for(k=0; k<N; k++){
				if(i > k)
					continue;
				C1[i*N + j] += A[i*N + k] * B[k*N + j];
			}
		}
	}

	for(i=0; i<N; i++){
		for(j=0; j<N; j++){
			C[i*N + j] = 0;
			for(k=0; k<N; k++){

				C[i*N + j] += C1[i*N + k] * B[j*N + k];

			}
		}
	}

	for(i=0; i<N; i++){
		for(j=0; j<N; j++){
			for(k=0; k<N; k++){
				if(i < k)
					continue;
				C[i*N + j] += A[k*N + i] * A[k*N + j];
			}
		}
	}
	free(C1);
	return C;
}
