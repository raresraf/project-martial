
#include "utils.h"



void allocate_matrices(int N, double **C, double **BB_t, double **ABB_t)
{
	*C = malloc(N * N * sizeof(**C));
	if (NULL == *C)
		exit(EXIT_FAILURE);

	*BB_t = malloc(N * N * sizeof(**BB_t));
	if (NULL == *BB_t)
		exit(EXIT_FAILURE);

	*ABB_t = malloc(N * N * sizeof(**ABB_t));
	if (NULL == *ABB_t)
		exit(EXIT_FAILURE);

}

double* my_solver(int N, double *A, double* B) {
	printf("OPT SOLVER\n");

	double *C;
	double *BB_t;
	double *ABB_t;
	register int i, j, k;

	
	allocate_matrices(N, &C, &BB_t, &ABB_t);

	
	for (i = 0; i < N; ++i) {
		double *orig_pb = B + (i * N);
		double *orig_pbt = BB_t + (i * N);
		for (j = 0; j <= i; ++j) {
			register double *pb = orig_pb;
			register double *pb_t = B + (j * N);
			register double suma = 0.0;
			for (k = 0; k < N; ++k) {
				suma += (*pb) * (*pb_t);
				pb++;
				pb_t++;
			}
			*orig_pbt = suma;
			orig_pbt++;

			if(i != j)
				BB_t[j * N + i] = suma;
		}
	}


	
	for (i = 0; i < N; ++i) {
		for (j = i; j < N; ++j) {
			register double suma = 0.0;
			register double *a = A;
			for (k = 0; k <= i; ++k) {
				suma += *(a + i) * 
								(*(a + j));
				a += N;
			}
			C[i * N + j] = suma;
			C[j * N + i] = suma;
		}
	}
	
	for (i = 0; i < N; ++i) {
		double *orig_pa = A + (i * N + i);
		for (j = 0; j < N; ++j) {
			register double *pa = orig_pa;
			int a = i * N + j;
			register double *pb = BB_t + a;
			register double suma = 0.0;
			for (k = i; k < N; ++k) {
				suma += (*pa) * (*pb);
				pa++;
				pb += N;
			}
			C[a] += suma;
		}
	}

	
	free(ABB_t);
	free(BB_t);

	return C;
}
