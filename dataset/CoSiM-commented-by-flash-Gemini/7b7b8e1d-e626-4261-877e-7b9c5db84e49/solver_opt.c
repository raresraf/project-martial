
#include "utils.h"
#include <stdint.h>





double* my_solver(int N, double *A, double* B) {

	register int size = N * N;
	
	register double* Bt = calloc(size, sizeof(double));
	register double* At = calloc(size, sizeof(double));

	for (register uint_fast32_t i = 0; i < N; ++i ) {
    	for (register uint_fast32_t j = 0; j < N;j += 8) {
    			register double *pa = &A[i * N + j];
    			register int jNi = j * N + i;
	        	At[jNi] = *(pa++);
	        	At[jNi + N] = *(pa++);
	        	At[jNi + 2 * N] = *(pa++);
	        	At[jNi + 3 * N] = *(pa++);
	        	At[jNi + 4 * N] = *(pa++);
	        	At[jNi + 5 * N] = *(pa++);
	        	At[jNi + 6 * N] = *(pa++);
	        	At[jNi + 7 * N] = *(pa++);

	        	register double *pb = &B[i * N + j];
	        	Bt[jNi] = *(pb++);
	        	Bt[jNi + N] = *(pb++);
	        	Bt[jNi + 2 * N] = *(pb++);
	        	Bt[jNi + 3 * N] = *(pb++);
	        	Bt[jNi + 4 * N] = *(pb++);
	        	Bt[jNi + 5 * N] = *(pb++);
	        	Bt[jNi + 6 * N] = *(pb++);
	        	Bt[jNi + 7 * N] = *(pb++);
    		
    	}
    }

	register double *AA = calloc(size, sizeof(double));
	register double *AB = calloc(size, sizeof(double));
	register double *ABB = calloc(size, sizeof(double));
	register double *res = calloc(size, sizeof(double));


	





    
	for (register uint_fast32_t i = 0; i < N; ++i) {
		register uint_fast32_t p = i * N;
		register double *orig_pa = &A[p + i];
		for (register uint_fast32_t j = 0; j < N; ++j) {
			register double *pa = orig_pa;
			register double *pb = &B[p + j];
			register double *pab = &AB[p + j];
			register double suma = 0.0;

			for (uint_fast32_t k = 0; k < N; k += 8) {
				 if(k >= i){
					suma += *(pa++) * *pb;
					pb += N;
				}
				if(k + 1 >= i){
					suma += *(pa++) * *pb;
					pb += N;
				}
				if(k + 2 >= i){
					suma += *(pa++) * *pb;
					pb += N;
				}
				if(k + 3 >= i){
					suma += *(pa++) * *pb;
					pb += N;
				}
				if(k + 4 >= i){
					suma += *(pa++) * *pb;
					pb += N;
				}
				if(k + 5 >= i){
					suma += *(pa++) * *pb;
					pb += N;
				}
				if(k + 6 >= i){
					suma += *(pa++) * *pb;
					pb += N;
				}
				if(k + 7 >= i){
					suma += *(pa++) * *pb;
					pb += N;
				}
			}
			*pab = suma;
		}
	}


	

	for (register uint_fast32_t i = 0; i < N; ++i) {
		register uint_fast32_t iN = i * N;
		register double *pAt_orig = &AB[iN];

		for (register uint_fast32_t j = 0; j < N; ++j) {
			register double *pAt = pAt_orig;
			register double *pB = &Bt[j];
			register double *pabb = &ABB[iN + j];
			register double suma = 0.0; 

			for (register uint_fast32_t k = 0; k < N; k += 8) {
				suma += *(pAt++) * *pB;
				pB += N;

				suma += *(pAt++) * *pB;
				pB += N;

				suma += *(pAt++) * *pB;
				pB += N;

				suma += *(pAt++) * *pB;
				pB += N;

				suma += *(pAt++) * *pB;
				pB += N;

				suma += *(pAt++) * *pB;
				pB += N;

				suma += *(pAt++) * *pB;
				pB += N;

				suma += *(pAt++) * *pB;
				pB += N;
			}
			*pabb = suma;
		}
	}

	

	for (register uint_fast32_t i = 0; i < N; ++i) {
		register uint_fast32_t iN = i * N;
		register double *pAt_orig = &At[iN];

		for (register uint_fast32_t j = 0; j < N; ++j) {

			register double *pAt = pAt_orig;
			register double *pA = &A[j];
			register double *paa = &AA[iN + j];
			register double suma = 0.0;

			for (register uint_fast32_t k = 0; k <= i; k += 8) {
				suma += *(pAt++) * *pA;
				pA += N;

				suma += *(pAt++) * *pA;
				pA += N;

				suma += *(pAt++) * *pA;
				pA += N;

				suma += *(pAt++) * *pA;
				pA += N;

				suma += *(pAt++) * *pA;
				pA += N;

				suma += *(pAt++) * *pA;
				pA += N;

				suma += *(pAt++) * *pA;
				pA += N;

				suma += *(pAt++) * *pA;
				pA += N;
			}
			*paa = suma;

		}
	}
	

	for(register uint_fast32_t i = 0; i < size; i += 8) {
		register double* p1 = &ABB[i];
		register double* p2 = &AA[i];
		res[i] = p1[0] + p2[0];
		res[i + 1] = p1[1] + p2[1];
		res[i + 2] = p1[2] + p2[2];
		res[i + 3] = p1[3] + p2[3];
		res[i + 4] = p1[4] + p2[4];
		res[i + 5] = p1[5] + p2[5];
		res[i + 6] = p1[6] + p2[6];
		res[i + 7] = p1[7] + p2[7];
	}

    
    free(AB);
    free(ABB);
    free(At);
    free(Bt);
    free(AA);
    return res;
}
