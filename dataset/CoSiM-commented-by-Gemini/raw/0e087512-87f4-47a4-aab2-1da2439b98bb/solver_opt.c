
#include "utils.h"

void alloc_matr(int N, double **C, double **AB, double **AA_t, double **ABB_t, double **A_t, double **B_t)
{
	*C = calloc(N * N, sizeof(**C));
	if (!*C)
		exit(EXIT_FAILURE);

	*AB = calloc(N * N, sizeof(**AB));
	if (!*AB)
		exit(EXIT_FAILURE);

	*AA_t = calloc(N * N, sizeof(**AA_t));
	if (!*AA_t)
		exit(EXIT_FAILURE);

	*ABB_t = calloc(N * N, sizeof(**ABB_t));
	if (!*ABB_t)
		exit(EXIT_FAILURE);

	*A_t = calloc(N * N, sizeof(**A_t));
	if (!*A_t)
		exit(EXIT_FAILURE);

	*B_t = calloc(N * N, sizeof(**B_t));
	if (!*B_t)
		exit(EXIT_FAILURE);
}

double* my_solver(int N, double *A, double* B) {
	printf("OPT SOLVER\n");

	register int i, j, k;

	double *AB = NULL;
	double *C = NULL;
	double *AA_t = NULL;
	double *ABB_t = NULL;

	double *A_t = NULL;
	double *B_t = NULL;

	alloc_matr(N, &C, &AB, &AA_t, &ABB_t, &A_t, &B_t);

	for (i = 0; i < N; i++) {
		register double *a = A_t + i;
		register double *b = B_t + i;
		
		register double *a_l = A + i * N;
		register double *b_l = B + i * N;

		for (j = 0; j < N; j++) {
			*a = *a_l;
			*b = *b_l;
			a += N;
			b += N;
			a_l++;
			b_l++;
		}
	}
	
	
	for (i = 0; i < N; ++i) {
		register double *AB_l = AB + i * N;
		
		register double *a_l = A + i * N;

		for (j = 0; j < N; ++j) {
			register double sum = 0.0;
			
			register double *A_el = a_l;
			register double *B_el = B_t + j * N;

			for (k = 0; k < N; ++k) {
				sum += *A_el * *B_el;
				B_el++;
				A_el++;
			}

			*AB_l = sum;
			++AB_l;
		}
	}

    
	for (i = 0; i < N; i++) {
		
		register double *ABB_t_l = ABB_t + i * N;
		
		register double *AB_l = AB + i * N;

		for (j = 0; j < N; j++) {
			register double sum = 0.0;
			
			register double *AB_el = AB_l;
			register double *B_t_el = B + j * N;

			for (k = 0; k < N; k++) {
				sum += *AB_el * *B_t_el;
				++AB_el;
				++B_t_el;
			}

			*ABB_t_l = sum;
			++ABB_t_l;
		}
	}

    
	for (i = 0; i < N; i++) {
		register double *AA_t_l = AA_t + i * N;
		
		register double *A_t_l = A_t + i * N;
		for (j = 0; j < N; j++) {
			register double sum = 0.0;
			register double *A_t_el = A_t_l;
			
			register double *a_el = A_t + j * N;

			for (k = 0; k < N; k++) {
				sum += *A_t_el * *a_el;
				++A_t_el;
				++a_el;
			}

			*AA_t_l = sum;
			++AA_t_l;
		}
	}
	
	
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			C[i * N + j] = ABB_t[i * N + j] + AA_t[i * N + j];
		}
	}

	free(AB);
	free(AA_t);
	free(ABB_t);
	free(A_t);
	free(B_t);

	return C;
}