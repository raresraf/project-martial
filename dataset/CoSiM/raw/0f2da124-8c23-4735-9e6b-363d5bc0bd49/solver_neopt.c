
#include "utils.h"



int min(int i, int j){
    return (i < j) ? i : j;
}


double* mult_At_A(int N, double* A, double* B) {
    int i, j, k;
    double *c;
    c = calloc(N * N, sizeof(double));
    
    for (i = 0; i < N; i++){
        for (j = 0; j < N; j++){
            for (k = 0; k < min(i + 1, j + 1); k++){
                c[i * N + j] += A[k * N + i] * B[k * N + j];
            }
        }
    }
    return c;
}


double* mult_mult1_Bt(int N, double* A, double* B) {
    int i, j, k;
    double *c;
    c = calloc(N * N, sizeof(double));
    
    for (i = 0; i < N; i++){
        for (j = 0; j < N; j++){
            for (k = 0; k < N; k++){
                c[i * N + j] += A[i * N + k] * B[j * N + k];
            }
        }
    }
    return c;
}


double* add_matrix(int N, double* A, double* B) {
    int i, j;
    double *c;
    c = calloc(N * N, sizeof(double));
    
    for (i = 0; i < N; i++){
        for(j = 0; j < N; j++) 
            c[i * N + j] += A[i * N + j] + B[i * N + j];
    }
    return c;
}


double* mult_A_B(int N, double* A, double* B) {
    int i, j, k;
    double *c;
    c = calloc(N * N, sizeof(double));
    
    for (i = 0; i < N; i++){
        for (j = 0; j < N; j++){
            for (k = i; k < N; k++){
                c[i * N + j] += A[i * N + k] * B[k * N + j];
            }
        }
    }
    return c;
}

void allocate_matrices(int N, double **C, double **AB, double **ABBt,
	double **AtA)
{
	*C = malloc(N * N * sizeof(**C));
	if (NULL == *C)
		exit(EXIT_FAILURE);

	*AB = calloc(N * N, sizeof(**AB));
	if (NULL == *AB)
		exit(EXIT_FAILURE);

	*ABBt = calloc(N * N, sizeof(**ABBt));
	if (NULL == *ABBt)
		exit(EXIT_FAILURE);

	*AtA = calloc(N * N, sizeof(**AtA));
	if (NULL == *AtA)
		exit(EXIT_FAILURE);
}

double* my_solver(int N, double *A, double *B) {
	double *AB;
    double *ABBt;
    double *AtA;
    double *C;

    AB = mult_A_B(N, A, B);
    ABBt = mult_mult1_Bt(N, AB, B);
    AtA = mult_At_A(N, A, A);
    C = add_matrix(N, ABBt, AtA);

    free(AB);
    free(ABBt);
    free(AtA);

    return C;
}
