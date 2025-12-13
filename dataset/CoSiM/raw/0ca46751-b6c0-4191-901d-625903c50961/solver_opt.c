
#include "utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MIN(a,b) (((a)<(b))?(a):(b))


double* my_solver(int N, double *A, double* B) {
	register double *rez;

	register double *prod11, *prod12;
	register double *prod2;

	
	register int i, j, k;
	register int m_size = N * N;
	register int d_size = sizeof(double);

	rez = calloc(m_size, d_size);
	if (rez == NULL)
		exit(EXIT_FAILURE);

	prod11 = calloc(m_size, d_size);
	if (prod11 == NULL)
		exit(EXIT_FAILURE);

	prod12 = calloc(m_size, d_size);
	if (prod12 == NULL)
		exit(EXIT_FAILURE);

	prod2 = calloc(m_size, d_size);
	if (prod2 == NULL)
		exit(EXIT_FAILURE);



	
	for (i = 0; i < N; i++) {
		
		register double *prod11_ptr = prod11 + i * N;
		for (j = 0; j < N; j++) {
			register double rez = 0;

			for (k = i; k < N; k++) {
				rez += A[i * N + k] * B[k * N + j];
			}
			*prod11_ptr = rez;

			prod11_ptr++;
		}
	}

	
	for (i = 0; i < N; i++) {
		register double *prod12_ptr = prod12 + i * N;

		for (j = 0; j < N; j++) {
			
			register double rez = 0;

			register double *B_ptr = B + j * N;
			register double *prod11_ptr = prod11 + i * N;

			for (k = 0; k < N; k++) {
				rez += *prod11_ptr * *B_ptr;

				B_ptr++;
				prod11_ptr++;
			}
			*prod12_ptr = rez;

			prod12_ptr++;
		}
	}

	
	for (i = 0; i < N; i++) {
		register double *prod2_ptr = prod2 + i * N;
		for (j = 0; j < N; j++) {
			register double rez = 0;
			for (k = 0; k <= MIN(i, j); k++) {
				rez += A[k * N + i] * A[k * N + j];

			}
			*prod2_ptr = rez;

			prod2_ptr++;
		}
	}


	
	for (i = 0; i < N; i++) {
		register double *rez_ptr = rez + i * N;
		register double *prod12_ptr = prod12 + i * N;
		register double *prod2_ptr = prod2 + i * N;

		for (j = 0; j < N; j++) {
			*rez_ptr = *prod12_ptr + *prod2_ptr;
			rez_ptr++;
			prod12_ptr++;
			prod2_ptr++;
		}


	}

	free(prod11);
	free(prod12);
	free(prod2);
	return rez;
}
