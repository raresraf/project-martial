/**
 * @file solver_opt.c
 * @brief Manually optimized matrix solver implementation.
 * Features register blocking, loop reordering, and cache-friendly data access patterns.
 */

/*
 * Module: solver_opt.c
 * Purpose: High-level matrix solver (optimized) with manually unrolled loops and pointer arithmetic.
 * Path: @raw/2f4487da-50d2-4a81-a14c-2397e860b9f3/solver_opt.c
 */
#include "utils.h"

/**
 * @brief Computes C = At * A + A * B * Bt.
 * Allocates memory dynamically and executes matrix operations.
 * @param N Matrix dimension.
 * @param A Input matrix A.
 * @param B Input matrix B.
 * @return Pointer to resulting matrix C.
 */
double* my_solver(int N, double *A, double* B) {
	printf("OPT SOLVER\n");

	double *C = (double *)malloc(N * N * sizeof(double));
    double *D = (double *)malloc(N * N * sizeof(double));
    register int i, j, k;

    
    /* @pre Loop bounds initialized. @invariant Iterates over assigned memory blocks, preserving data locality where possible. */
    for (i = 0; i < N; i++) {
    	register double *orig_pb1 = &B[i * N]; /* Non-obvious pointer arithmetic/dereference for optimized memory access */
        /* @pre Loop bounds initialized. @invariant Iterates over assigned memory blocks, preserving data locality where possible. */
        for (j = i; j < N; j++) {
            register double aux = 0;
            register double *pb1 = orig_pb1; /* Non-obvious pointer arithmetic/dereference for optimized memory access */
            register double *pb2 = &B[j * N]; /* Non-obvious pointer arithmetic/dereference for optimized memory access */

            /* @pre Loop bounds initialized. @invariant Iterates over assigned memory blocks, preserving data locality where possible. */
            for (k = 0; k < N; k++) {
                aux += *pb1 * *pb2; /* Non-obvious pointer arithmetic/dereference for optimized memory access */
                pb1++;
                pb2++;
            }

            C[i * N + j] = aux;
            /* @pre Conditional evaluation. @invariant Taken branch maintains control flow invariants. */
            if (i != j)
                C[j * N + i] = aux;

        }
    }

    
    /* @pre Loop bounds initialized. @invariant Iterates over assigned memory blocks, preserving data locality where possible. */
    for (i = 0; i < N; i++) {
    	register double *orig_pa = &A[i * (N + 1)]; /* Non-obvious pointer arithmetic/dereference for optimized memory access */
        /* @pre Loop bounds initialized. @invariant Iterates over assigned memory blocks, preserving data locality where possible. */
        for (j = 0; j < N; j++) {
            register double aux = 0;
            register double *pa = orig_pa; /* Non-obvious pointer arithmetic/dereference for optimized memory access */
            register double *pc = &C[j * N + i]; /* Non-obvious pointer arithmetic/dereference for optimized memory access */

            /* @pre Loop bounds initialized. @invariant Iterates over assigned memory blocks, preserving data locality where possible. */
            for (k = i; k < N; k++) {
                aux += *pa * *pc; /* Non-obvious pointer arithmetic/dereference for optimized memory access */
                pa++;
                pc++;
            }

            D[i * N + j] = aux;
        }
    }

    
    /* @pre Loop bounds initialized. @invariant Iterates over assigned memory blocks, preserving data locality where possible. */
    for (i = 0; i < N; i++) {
    	register double *orig_pa = &A[i]; /* Non-obvious pointer arithmetic/dereference for optimized memory access */
        /* @pre Loop bounds initialized. @invariant Iterates over assigned memory blocks, preserving data locality where possible. */
        for (j = i; j < N; j++) {
            register double aux = 0;
            register double *pa1 = orig_pa; /* Non-obvious pointer arithmetic/dereference for optimized memory access */
            register double *pa2 = &A[j]; /* Non-obvious pointer arithmetic/dereference for optimized memory access */

            /* @pre Loop bounds initialized. @invariant Iterates over assigned memory blocks, preserving data locality where possible. */
            for (k = 0; k <= i; k++) {
                aux += *pa1 * *pa2; /* Non-obvious pointer arithmetic/dereference for optimized memory access */
                pa1 += N;
                pa2 += N;
            }

            /* @pre Conditional evaluation. @invariant Taken branch maintains control flow invariants. */
            if (i != j) {
                D[j * N + i] += aux;
            }

            D[i * N + j] += aux;
        }
    }

    free(C);
	return D;	
}
