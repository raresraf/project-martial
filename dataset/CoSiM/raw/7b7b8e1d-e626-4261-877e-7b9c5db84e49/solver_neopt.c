
#include "utils.h"



double* my_solver(int N, double *A, double *B) {

    
    double *res = calloc(N * N, sizeof(double));
    double *AB = calloc(N * N, sizeof(double));
    double *ABB = calloc(N * N, sizeof(double));
    double *AA = calloc(N * N, sizeof(double));

    for (int i = 0; i < N; i++)
		for (int j = 0; j < N; j++) {
		    double suma = 0.0;
		    for (int k = 0; k < N; k++) {
				if(k >= i)
					suma += A[i * N + k] * B[k * N + j];
		    }
		    AB[i * N + j] = suma;
		}

	int k = 0;
	
	double *Bt = calloc(N * N, sizeof(double));

    for(int i = 0; i < N; i++){
		for (int j = 0; j < N; j++){
            Bt[k++] = B[j * N + i];
        }
    }

	for (int i = 0; i < N; i++)
		for (int j = 0; j < N; j++) {
			double suma = 0.0;
			for (int k = 0; k < N; k++) {
				suma += AB[i * N + k] * Bt[k * N + j];
			}
			ABB[i * N + j] = suma;
		}

	

	double *At = calloc(N * N, sizeof(double));

	k = 0;

	for(int i = 0; i < N; i++){
        for (int j = 0; j < N; j++){
            At[k++] = A[j * N + i];
        }
    }

    for (int i = 0; i < N; i++)
		for (int j = 0; j < N; j++) {
			double suma = 0.0;
			for (int k = 0; k < N; k++) {
				if(k <= i)
					suma += At[i * N + k] * A[k * N + j];
			}
			AA[i * N + j] = suma;
		}
	for(int i = 0; i < N * N; i++) {
		res[i] = AA[i] + ABB[i];
	}


	free(AA);
	free(AB);
	free(ABB);
	free(At);
	free(Bt);

	

	return res;
}
