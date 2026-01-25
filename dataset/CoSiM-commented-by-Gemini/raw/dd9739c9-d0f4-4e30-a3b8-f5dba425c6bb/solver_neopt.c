
#include "utils.h"





void add(double* result, double* A, double* B, int N) {
    for (int i = 0; i < N; i++) {
        for(int j = 0; j < N; j++) {
            result[i * N + j] = A[i * N + j] + B[i * N + j];
		}
    }
    
};

void prodTriSub(double* result, double* A, double* B, int N) {
	for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            double aux = 0;
            for (int k = 0; k < N; k++){
				if(k >= i) {
                	aux = aux + A[i * N + k] * B[k * N + j];
				}
            }
            result[i * N + j] = aux;
        }
    }
}


void prodTriLow(double* result, double* A, double* B, int N) {
	for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            double aux = 0;
            for (int k = 0; k < N; k++){
				if(k <= i) {
                	aux = aux + A[i * N + k] * B[k * N + j];
				}
            }
            result[i * N + j] = aux;
        }
    }
}



void prod(double* result, double* A, double* B, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            double aux = 0;
            for (int k = 0; k < N; k++){
                aux = aux + A[i * N + k] * B[k * N + j];
            }
            result[i * N + j] = aux;
        }
    }
}


double* my_solver(int N, double *A, double* B) {
	double* C = (double*) malloc(N * N * sizeof(double));
	double* At = (double*) malloc(N * N * sizeof(double));
	double* Bt = (double*) malloc(N * N * sizeof(double));

	
	for(int i = 0; i < N; i++) {
		for(int j = 0; j < N; j++) {
			At[i * N + j] = A[i + N * j];
			Bt[i * N + j] = B[i + N * j];
		}
	}


	

	double* X = (double*) malloc(N * N * sizeof(double));
	prodTriSub(X, A, B, N);

	
	double* Y = (double*) malloc(N * N * sizeof(double));
	prod(Y, X, Bt, N);

	
	double* Z = (double*) malloc(N * N * sizeof(double));
	prodTriLow(Z, At, A, N);


	
	add(C, Y, Z, N);

	
	free(At);
	free(Bt);
	free(X);
	free(Z);
	free(Y);
	return C;
}
