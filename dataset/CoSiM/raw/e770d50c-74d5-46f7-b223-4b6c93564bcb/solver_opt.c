
#include "utils.h"
#include <stdint.h>

double* multiply(uint_fast32_t N, double *A, double *B) {
	double* res = (double*)calloc(N*N, sizeof(double));
		for (register uint_fast32_t i = 0; i < N; i++) {
		register uint_fast32_t p = i * N;
		uint_fast32_t indexA = p +i;
			 indexA = i*N;
			
		 
		register double *orig_pa = &A[indexA];
		for (register uint_fast32_t j = 0; j < N; j++) {
					 uint_fast32_t indexB = p +j;

			 indexB = j;
			
		 
			register double *pa = orig_pa;
			register double *pb = &B[indexB] ;
			register double sum = 0.0;
			for (uint_fast32_t k = 0; k < N; k+=8) {
				 
					sum += *pa * *pb;
					pa++;
					pb += N;
				
			
					sum += *pa * *pb;
					pa++;
					pb += N;
				
				
					sum += *pa * *pb;
					pa++;
					pb += N;
				
				
					sum += *pa * *pb;
					pa++;
					pb += N;
				
					sum += *pa * *pb;
					pa++;
					pb += N;
				
					sum += *pa * *pb;
					pa++;
					pb += N;
				
					sum += *pa * *pb;
					pa++;
					pb += N;
				
					sum += *pa * *pb;
					pa++;
					pb += N;
				

				
			}
			res[p + j] = sum;
		}
	}
	return res;

}

double* multiplyAux(uint_fast32_t N, double *A, double *B) {
	double* res = (double*)calloc(N*N, sizeof(double));
		for (register uint_fast32_t i = 0; i < N; i++) {
		register uint_fast32_t p = i * N;
		uint_fast32_t indexA = p +i;

		register double *orig_pa = &A[indexA];
		for (register uint_fast32_t j = 0; j < N; j++) {
					 uint_fast32_t indexB = p +j;

			register double *pa = orig_pa;
			register double *pb = &B[indexB] ;
			register double sum = 0.0;
			for (uint_fast32_t k = 0; k < N; k+=8) {
				 if(k >= i ){
					sum += *pa * *pb;
					pa++;
					pb += N;
				}
				if(k + 1 >= i ){
					sum += *pa * *pb;
					pa++;
					pb += N;
				}
				if(k + 2>= i ){
					sum += *pa * *pb;
					pa++;
					pb += N;
				}
				if(k + 3>= i ){
					sum += *pa * *pb;
					pa++;
					pb += N;
				}
				if(k + 4>= i ){
					sum += *pa * *pb;
					pa++;
					pb += N;
				}
				if(k + 5>= i ){
					sum += *pa * *pb;
					pa++;
					pb += N;
				}
				if(k + 6>= i ){
					sum += *pa * *pb;
					pa++;
					pb += N;
				}
				if(k + 7>= i ){
					sum += *pa * *pb;
					pa++;
					pb += N;
				}

				
			}
			res[p + j] = sum;
		}
	}
	return res;

}

double* add(uint_fast32_t N, double *A, double *B) {
	double* res = (double*)calloc(N*N, sizeof(double));
        for (register uint_fast32_t j = 0; j < N * N; j+=8) {
            res[j] = A[j] + B[j];

            res[j + 1] = A[j + 1] + B[j + 1];

            res[j + 2] = A[j + 2] + B[j + 2];

            res[j + 3] = A[j + 3] + B[j + 3];

            res[j + 4] = A[j + 4] + B[j + 4];

            res[j + 5] = A[j + 5] + B[j + 5];

            res[j + 6] = A[j + 6] + B[j + 6];

            res[j + 7] = A[j + 7] + B[j + 7];
        }
	return res;

}

double* transpose(uint_fast32_t N, double *A) {
	double* res = (double*)calloc(N*N, sizeof(double));
	for (register uint_fast32_t i = 0; i < N; i++) {
        for (register uint_fast32_t j = 0; j < N; j+=8) {
          register uint_fast32_t index1 = i*N+j;
          register uint_fast32_t index2 = j*N+i;
            res[index2] = A[index1];

			index1 = i*N+j+1;
            index2 = (j+1)*N+i;
            res[index2] = A[index1];

			index1 = i*N+j+2;
            index2 = (j+2)*N+i;
            res[index2] = A[index1];

			index1 = i*N+j+3;
            index2 = (j+3)*N+i;
            res[index2] = A[index1];

			index1 = i*N+j+4;
            index2 = (j+4)*N+i;
            res[index2] = A[index1];

			index1 = i*N+j+5;
            index2 = (j+5)*N+i;
            res[index2] = A[index1];

			index1 = i*N+j+6;
            index2 = (j+6)*N+i;
            res[index2] = A[index1];

			index1 = i*N+j+7;
            index2 = (j+7)*N+i;
            res[index2] = A[index1];
        }
    }
	return res;

}


double* my_solver(int N, double* A, double* B) {
	double* a_trans = transpose(N, A);
	double* b_trans = transpose(N, B);

	double* a_trans_mult_a = multiply(N, a_trans, A);
	double* a_mult_b = multiplyAux(N, A, B);
	double* a_mult_b_mult_trans = multiply(N, a_mult_b, b_trans);

	double* final = add(N, a_mult_b_mult_trans, a_trans_mult_a);

	free(a_trans);
	free(b_trans);
	free(a_trans_mult_a);
	free(a_mult_b);
	free(a_mult_b_mult_trans);

	

	return final;
	
}