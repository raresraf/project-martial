
#include "utils.h"


double* my_solver(int N, double *A, double* B) {
	double *C;
	double *A_t;
	double *B_t;
	double *AB;
	register int i, j, k;
	register double *A_t_ptr, *B_t_ptr;
	register double *A_ptr, *B_ptr;
	register double *A_tAptr;
	register double *pa, *pb, result;
	register double *AB_ptr;
	register double *C_ptr;

	A_t = (double *)calloc(N * N, sizeof(double));
	if (NULL == A_t)
		exit(EXIT_FAILURE);
	
	B_t = (double *)calloc(N * N, sizeof(double));
	if (NULL == B_t)
		exit(EXIT_FAILURE);
	
	AB = (double *)calloc(N * N, sizeof(double));
	if (NULL == AB)
		exit(EXIT_FAILURE);
	
	C = (double *)calloc(N * N, sizeof(double));
	if (NULL == C)
		exit(EXIT_FAILURE);

	for (i = 0; i != N; ++i) {
		A_t_ptr = A_t + i;  
		B_t_ptr = B_t + i;  

		A_ptr = A + i * N;  
		B_ptr = B + i * N;  

		for (j = 0; j != N; ++j) {
			*A_t_ptr = *A_ptr;
			*B_t_ptr = *B_ptr;
			A_t_ptr += N;
			B_t_ptr += N;
			++A_ptr;
			++B_ptr;
		}
	}

	
	for (i = 0; i != N; ++i) {
		
		A_tAptr = C + i * N;

		
		A_t_ptr = A_t + i * N;

		for (j = 0; j != N; ++j) {
			pa = A_t_ptr;
			pb = A_t + j * N;
			result = 0;

			for (k = 0; k <= i; ++k) {
				result += *pa * *pb;
				++pa;
				++pb;
			}

			*A_tAptr = result;
			++A_tAptr; 
		}
	}

	

	for (i = 0; i != N; ++i) {
		
		AB_ptr = AB + i * N;

		
		A_ptr = A + i * N;

		for (j = 0; j != N; ++j) {
			pa = A_ptr + i;
			pb = B_t + j * N + i;
			result = 0;

			for (k = i; k < N; ++k) {
				result += *pa * *pb;
				++pa;
				++pb;
			}

			*AB_ptr = result;
			++AB_ptr; 
		}
	}

	
	for (i = 0; i != N; ++i) {
		
		C_ptr = C + i * N;

		
		AB_ptr = AB + i * N;

		for (j = 0; j != N; ++j) {
			pa = AB_ptr;
			pb = B + j * N;
			result = 0;

			for (k = 0; k != N; ++k) {
				result += *pa * *pb;
				++pa;
				++pb;
			}

			*C_ptr += result;
			++C_ptr; 
		}
	}
	free(A_t);
	free(B_t);
	free(AB);
	return C;	
}