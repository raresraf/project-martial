
#include "utils.h"



double* transpose(int N, double* A) {
	double* At = malloc(N * N * sizeof(double));

	register size_t i, j;
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			At[i*N + j] = A[j*N + i];
		}
	}
	return At;
}

double* inmultire(int N, double *A, double* B) {
	double* M = malloc( N * N * sizeof(double));

	register size_t i, j, k;
	
	for (i = 0; i < N ; i++) {
		register double* org_pa = &A[N*i];
		register double* org_M = &M[N*i];

		for (j = 0; j < N ; j++) {
		    register double *pa = org_pa;
			register double* pb = &B[N*j];
			register double calcul = 0;
			
			for (k = 0; k < N; k++) {
				calcul += *(pa) * (*pb);
				pa++;
				pb++;
			}

			*(org_M) = calcul;
			org_M++;
		}
	}

	return M;
}

double* inmultireSuperiorTriunghiulara(int N, double* A, double* B) {
	double* M = calloc(sizeof(double), N * N);

	register size_t i, j, k;

	for (i = 0; i < N ; i++) {
		register double* org_pa = &A[N*i];
		register double* org_M = &M[N*i];

		for (j = 0; j < N ; j++) {
		    register double *pa = org_pa + i;
			register double* pb = &B[N*j + i];
			register double calcul = 0;

			for (k = i; k < N; k++){
				calcul += *(pa) * (*pb);
				pa++;
				pb++;
			}
			*(org_M) = calcul;
			org_M++;
		}
	}

	return M;
}

double* inmultireLowerXUpper(int N, double* A, double* B) {
	double* M = calloc(sizeof(double), N * N);

	register size_t i, j, k;
	for (i = 0; i < N ; i++) {
		register double* org_pa = &A[N*i];
		register double* org_M = &M[N*i];

		for (j = 0; j < N ; j++) {
		    register double *pa = org_pa;
			register double* pb = &B[N*j];
			register double calcul = 0;

			for (k = 0; k < j + 1; k++){
				calcul += *(pa) * (*pb);
				pa++;
				pb++;
			}
			*(org_M) = calcul;
			org_M++;
		}
	}

	return M;
}

double* adunare(int N, double *A, double* B) {
	double* M = calloc(sizeof(double), N * N);
	
	register size_t i, j;
	for (i = 0; i < N ; i++) {
		for (j = 0; j < N ; j++)
			M[i*N + j] = A[i*N + j] + B[i*N + j]; 
	}
	return M;
}

double* my_solver(int N, double *A, double* B) {
	double* B_transpose = transpose(N, B);
	double* AxB = inmultireSuperiorTriunghiulara(N, A, B_transpose);
	double* firstPart = inmultire(N, AxB, B);
	double* A_transpose = transpose(N, A);
	double* secondPart = inmultireLowerXUpper(N, A_transpose, A_transpose);

	double* C = adunare(N, firstPart, secondPart);

	free(AxB);
	free(B_transpose);
	free(firstPart);
	free(secondPart);
	free(A_transpose);

	return C;
}