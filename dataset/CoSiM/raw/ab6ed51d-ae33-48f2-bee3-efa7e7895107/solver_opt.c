
#include "utils.h"
#include <string.h>


double* my_solver(int N, double *A, double* B) {
	printf("OPT SOLVER\n");

	

	register int i,j,k;
	
	double *C = malloc(sizeof(double) * N * N);
	if (!C) 
		return NULL;

	double *AB = malloc(sizeof(double) * N * N);
	if (!AB)
		return NULL;

	double *BBt = malloc(sizeof(double) * N * N);
	if (!BBt)
		return NULL;

	double *AtA = malloc(sizeof(double) * N * N);
	if (!AtA)
		return NULL;

	
	for (i = 0; i < N; i++) {
		register double	*a = A + i * N;
		for (j = 0; j < N; j++) {
			register double *aa = a + i;
			register double *b = B + j + i * N;
			register double	sum = 0.0;
			for (k = i; k < N; k++) {
				sum += *aa * *b;
				aa++;
				b += N;
			}
			AB[i * N + j] = sum;
		}
	}

	
	for (i = 0; i < N; i++) {
		register double	*a = AB + i * N;
		for (j = 0; j < N; j++) {
			register double	*aa = a;
			register double	*b = B + j * N;
			register double	sum = 0.0;
			for (k = 0; k < N; k++) {
				sum += *aa * *b;
				aa++;
				b++;
			}
			BBt[i * N + j] = sum;
		}
	}

	
	for (i = 0; i < N; i++) {
		register double	*a = A + i;
		for (j = 0; j < N; j++) {
			
			register double	sum = 0.0;
			for (k = 0; k < N; k++) {
				register double	*aa = a + k * N;
				register double	*b = A + j + k * N;
				sum += *aa * *b;
				aa+=N;
				b+=N;
			}
			AtA[i * N + j] = sum;
		}
	}

	for (i = 0; i < N; i++) {
		register double	*a = BBt + i * N;
		register double	*aa = AtA + i * N;
		for (j = 0; j < N; j++) {
			C[i * N + j] = *aa + *a;
			aa++;
			a++;
		}
	}

	free(BBt);
	free(AtA);
	free(AB);

	return C;	
}
