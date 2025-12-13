
#include "utils.h"



void transpose_matrix(int N,double *initial_matrix, double *transpose)
{
	register int i,j;
	for (i = 0; i < N; i++)
  		for (j = 0; j < N; j++) {
    		transpose[j * N + i] = initial_matrix[i * N + j];
  }
}

double* multiply_matrix(int N, double *A, double* B)
{
	register int i,j,k;
	double*  result = calloc(N*N ,sizeof(result));
	for (i = 0 ; i < N ; i++)
	{
		register double *line = &A[i * N];
		for (j = 0 ; j < N ; j++) {
			register double *curent_line = line;
    		register double *curent_col = &B[j];
			register double suma = 0.0;	
			for(k = 0 ; k < N ; k ++) 
			{
				suma +=*curent_line * *curent_col;
				curent_line++;
				curent_col += N;
			}
			result[i * N + j] = suma;
		}
	}
	
	return result;
}

double* multiply_matrix_inferior(int N, double *A, double* B)
{
	register int i,j,k;
	double*  result = calloc(N*N , sizeof(result));
	for (i = 0 ; i < N ; i++) {
		register double *line = &A[i * N];
		for (j = 0 ; j < N ; j++) {
			register double *curent_line = line;
    		register double *curent_col = &B[j];
			register double suma = 0.0;
			for(k = 0 ; k <= i ; k++) {
				suma +=*curent_line * *curent_col;
				curent_line++;
				curent_col += N;
			}
			result[i * N + j] = suma;
		}
	}
	return result;
}

double* multiply_matrix_superior(int N, double *A, double* B)
{
	register int i,j,k;
	double*  result = calloc(N*N , sizeof(result));
	for (i = 0 ; i < N ; i++) {
		for (j = 0 ; j < N ; j++){
			register double suma = 0.0;	
			for(k = i ; k < N ; k++)
			{
				suma += A[i * N + k] * B[k * N + j];
			}
			result[i * N + j] = suma;
		}
	}
	
	return result;
}

double* add_matrix(int N, double *A, double* B)
{
	register int i,j;
	double*  result = calloc(N*N , sizeof(result));
	for (i = 0; i < N; i++)
    	for (j = 0; j < N; j++) 
      		result[i * N + j] = A[i * N + j] +  B[i * N + j];

	return result;
    
}

double* my_solver(int N, double *A, double* B) {
	printf("OPT SOLVER\n");
	double * result;
	double * resultAXB;
	double * resultXTranspose;
	double * A_transposeXA;
	double * A_transpose = (double *) calloc(N*N , sizeof(double));
	double * B_transpose = (double *) calloc(N*N , sizeof(double));

	transpose_matrix(N,A,A_transpose);
	transpose_matrix(N,B,B_transpose);

	
	resultAXB = multiply_matrix_superior(N,A,B);
	resultXTranspose = multiply_matrix(N,resultAXB,B_transpose);
	
	A_transposeXA = multiply_matrix_inferior(N,A_transpose,A);


	result = add_matrix(N,resultXTranspose,A_transposeXA);

	free(A_transposeXA);
	free(B_transpose);
	free(A_transpose);
	free(resultXTranspose);
	free(resultAXB);

	return result;	
}
