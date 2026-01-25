
#include "utils.h"


double* my_solver(int N, double *A, double* B) {
	printf("OPT SOLVER\n");
	
	
	double *AB = calloc(N * N, sizeof(double));
	double *ABBt = malloc(N * N * sizeof(double));
	double *AtA = calloc(N * N, sizeof(double));
	double *C = malloc(N * N * sizeof(double));

	if (AB == NULL || ABBt == NULL || AtA == NULL || C == NULL){
		fprintf(stderr, "malloc error\n");
		exit(EXIT_FAILURE);
	}

	
	for (int i = 0; i < N; i++) {
		double *copy_res = &(AB[i * N]);
		*copy_res = 0;
		for (int k = 0; k < N; k++) {
			double *line = &(A[i * N + k]);
			double *col = &(B[k * N]);
			double *res = copy_res;
			
			if (i <= k) {
				for (int j = 0; j < N; j += 4) {
					
					*res += *line * *col;
					col++;
					res++;
					*res += *line * *col;
					col++;
					res++;
					*res += *line * *col;
					col++;
					res++;
					*res += *line * *col;
					col++;
					res++;
				}
			}
		}
	}

	
	for (int i = 0; i < N; i++) {
		double *copy_line = &(AB[i * N]);
		for (int j = 0; j < N; j++) {
			double *line = copy_line;
			double *col = &(B[j * N]);
			register double sum = 0.0;
			for (int k = 0; k < N; k += 4) {
				
				sum += *line * *col;
				line++;
				col++;
				sum += *line * *col;
				line++;
				col++;
				sum += *line * *col;
				line++;
				col++;
				sum += *line * *col;
				line++;
				col++;
			}
			ABBt[i * N + j] = sum;
		}
	}

	
	for (int k = 0; k < N; k++) {
		double *copy_col = &(A[k * N]);
		for (int i = 0; i < N; i++) {
			double *res = &(AtA[i * N]);
			double *line = &(A[k * N + i]);
			double *col = copy_col;

			if (i < k) {
				continue;
			}
			for (int j = 0; j < N; j++) {
				
				if (k <= j) {
					
					*res += *line * *col;
				}
				col++;
				res++;
			}
		}
	}

	
	double *a = &(ABBt[0]);
	double *b = &(AtA[0]);
	double *c = &(C[0]);
	for (int i = 0; i < N * N; i += 4) {
		
		*c = *a + *b;
		a++;
		b++;
		c++;
		*c = *a + *b;
		a++;
		b++;
		c++;
		*c = *a + *b;
		a++;
		b++;
		c++;
		*c = *a + *b;
		a++;
		b++;
		c++;
	}

	
	free(AB);
	free(ABBt);
	free(AtA);

	return C;
}
