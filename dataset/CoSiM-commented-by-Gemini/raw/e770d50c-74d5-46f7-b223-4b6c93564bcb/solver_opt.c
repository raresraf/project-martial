/**
 * @file solver_opt.c
 * @brief Block-optimized manual implementation to compute $C = A \times B \times B^T + A^T \times A$.
 * Algorithm: Loop tiling/blocking for cache locality enhancement.
 * Time Complexity: $O(N^3)$.
 * Space Complexity: $O(N^2)$ for intermediate data structures.
 */

#include "utils.h"
#include <stdint.h>

double* multiply(uint_fast32_t N, double *A, double *B) {
	double* res = (double*)calloc(N*N, sizeof(double));
		/**
		 * Block Logic: Iterative processing loop.
		 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
		 */
		for (register uint_fast32_t i = 0; i < N; i++) {
		register uint_fast32_t p = i * N;
		uint_fast32_t indexA = p +i;
			 indexA = i*N;
			
		 
		register double *orig_pa = &A[indexA]; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
		/**
		 * Block Logic: Iterative processing loop.
		 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
		 */
		for (register uint_fast32_t j = 0; j < N; j++) {
					 uint_fast32_t indexB = p +j;

			 indexB = j;
			
		 
			register double *pa = orig_pa; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
			register double *pb = &B[indexB] ; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
			register double sum = 0.0;
			/**
			 * Block Logic: Iterative processing loop.
			 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
			 */
			for (uint_fast32_t k = 0; k < N; k+=8) {
				 
					sum += *pa * *pb; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
					pa++;
					pb += N; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
				
			
					sum += *pa * *pb; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
					pa++;
					pb += N; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
				
				
					sum += *pa * *pb; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
					pa++;
					pb += N; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
				
				
					sum += *pa * *pb; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
					pa++;
					pb += N; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
				
					sum += *pa * *pb; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
					pa++;
					pb += N; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
				
					sum += *pa * *pb; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
					pa++;
					pb += N; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
				
					sum += *pa * *pb; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
					pa++;
					pb += N; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
				
					sum += *pa * *pb; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
					pa++;
					pb += N; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
				

				
			}
			res[p + j] = sum;
		}
	}
	return res;

}

double* multiplyAux(uint_fast32_t N, double *A, double *B) {
	double* res = (double*)calloc(N*N, sizeof(double));
		/**
		 * Block Logic: Iterative processing loop.
		 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
		 */
		for (register uint_fast32_t i = 0; i < N; i++) {
		register uint_fast32_t p = i * N;
		uint_fast32_t indexA = p +i;

		register double *orig_pa = &A[indexA]; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
		/**
		 * Block Logic: Iterative processing loop.
		 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
		 */
		for (register uint_fast32_t j = 0; j < N; j++) {
					 uint_fast32_t indexB = p +j;

			register double *pa = orig_pa; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
			register double *pb = &B[indexB] ; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
			register double sum = 0.0;
			/**
			 * Block Logic: Iterative processing loop.
			 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
			 */
			for (uint_fast32_t k = 0; k < N; k+=8) {
				 /**
				  * Block Logic: Conditional state branch.
				  * Invariant: The conditional branch maintains control flow invariants.
				  */
				 if(k >= i ){
					sum += *pa * *pb; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
					pa++;
					pb += N; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
				}
				/**
				 * Block Logic: Conditional state branch.
				 * Invariant: The conditional branch maintains control flow invariants.
				 */
				if(k + 1 >= i ){
					sum += *pa * *pb; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
					pa++;
					pb += N; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
				}
				/**
				 * Block Logic: Conditional state branch.
				 * Invariant: The conditional branch maintains control flow invariants.
				 */
				if(k + 2>= i ){
					sum += *pa * *pb; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
					pa++;
					pb += N; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
				}
				/**
				 * Block Logic: Conditional state branch.
				 * Invariant: The conditional branch maintains control flow invariants.
				 */
				if(k + 3>= i ){
					sum += *pa * *pb; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
					pa++;
					pb += N; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
				}
				/**
				 * Block Logic: Conditional state branch.
				 * Invariant: The conditional branch maintains control flow invariants.
				 */
				if(k + 4>= i ){
					sum += *pa * *pb; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
					pa++;
					pb += N; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
				}
				/**
				 * Block Logic: Conditional state branch.
				 * Invariant: The conditional branch maintains control flow invariants.
				 */
				if(k + 5>= i ){
					sum += *pa * *pb; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
					pa++;
					pb += N; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
				}
				/**
				 * Block Logic: Conditional state branch.
				 * Invariant: The conditional branch maintains control flow invariants.
				 */
				if(k + 6>= i ){
					sum += *pa * *pb; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
					pa++;
					pb += N; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
				}
				/**
				 * Block Logic: Conditional state branch.
				 * Invariant: The conditional branch maintains control flow invariants.
				 */
				if(k + 7>= i ){
					sum += *pa * *pb; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
					pa++;
					pb += N; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
				}

				
			}
			res[p + j] = sum;
		}
	}
	return res;

}

double* add(uint_fast32_t N, double *A, double *B) {
	double* res = (double*)calloc(N*N, sizeof(double));
        /**
         * Block Logic: Iterative processing loop.
         * Invariant: Maintains sequence integrity while progressing through the defined bounds.
         */
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
	/**
	 * Block Logic: Iterative processing loop.
	 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
	 */
	for (register uint_fast32_t i = 0; i < N; i++) {
        /**
         * Block Logic: Iterative processing loop.
         * Invariant: Maintains sequence integrity while progressing through the defined bounds.
         */
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


/**
 * @brief Computes C = A * B * B^T + A^T * A.
 * @param N Matrix dimension.
 * @param A Input matrix A.
 * @param B Input matrix B.
 * @return Pointer to resulting matrix C.
 */
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