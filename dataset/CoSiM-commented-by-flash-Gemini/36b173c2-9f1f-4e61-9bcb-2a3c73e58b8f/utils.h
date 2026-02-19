/**
 * @file utils.h
 * @brief Provides utility functions, data structures, and macros for matrix solver implementations.
 *
 * This header file defines common types, helper macros, and function prototypes
 * used by various matrix solver implementations (e.g., optimized, non-optimized, BLAS-based).
 * It includes functionalities for data generation, input/output, and timing.
 */

#include <stdio.h>    // Functional Utility: For standard input/output operations (e.g., printf, sscanf).
#include <stdlib.h>   // Functional Utility: For general utilities like memory allocation (malloc, free, calloc) and exit.
#include <sys/time.h> // Functional Utility: For timing functions like gettimeofday.

/**
 * @typedef Solver
 * @brief Defines a function pointer type for matrix solver functions.
 *
 * A `Solver` function takes the matrix dimension `N` and two N x N double-precision
 * matrices `A` and `B` as input, and returns a pointer to a newly allocated N x N
 * double-precision matrix containing the result.
 *
 * @param N (int): The dimension of the square matrices.
 * @param A (double*): Pointer to the first input matrix.
 * @param B (double*): Pointer to the second input matrix.
 * @return (double*): Pointer to the result matrix.
 */
typedef double* (*Solver)(int, double *, double*);

/**
 * @brief Macro to generate a random double-precision floating-point number within a specified range.
 *
 * Generates a random double between `-limit` and `+limit`.
 *
 * @param limit (double): The absolute maximum value for the generated random number.
 * @return (double): A random double value in the range [-limit, +limit].
 */
#define get_rand_double(limit) ((((double)rand()) / RAND_MAX) * (2 * limit) - limit)

/**
 * @struct test
 * @brief Structure to hold parameters for a single test case.
 *
 * @var test::seed (int): Seed for random number generation to ensure reproducible test data.
 * @var test::N (int): Dimension of the square matrices (N x N) for the test.
 * @var test::output_save_file (char[100]): File path to save the output matrix.
 */
struct test {
	int seed;
	int N;
	char output_save_file[100];
};

/**
 * @brief Function prototype for the matrix solver.
 *
 * This function performs a specific matrix computation, defined by the solver implementation.
 * It is expected to take two N x N matrices and return a new N x N result matrix.
 *
 * @param N (int): The dimension of the square matrices.
 * @param A (double*): Pointer to the first input matrix.
 * @param B (double*): Pointer to the second input matrix.
 * @return (double*): Pointer to the result matrix.
 */
double* my_solver(int, double *, double *);

/**
 * @brief Runs a single test case with a specified solver.
 *
 * This function orchestrates the execution of a single test, including
 * data generation, calling the solver, and potentially measuring performance.
 *
 * @param test_case (struct test): The test case parameters.
 * @param solver_func (Solver): Pointer to the solver function to be tested.
 * @param execution_time (float*): Pointer to a float where the execution time will be stored.
 * @return (int): 0 on success, non-zero on error.
 */
int run_test(struct test, Solver, float *);

/**
 * @brief Frees dynamically allocated memory for matrix data.
 *
 * @param data (double**): A pointer to a pointer to the matrix data to be freed.
 */
void free_data(double **);

/**
 * @brief Generates random double-precision floating-point matrix data for a test.
 *
 * Allocates and fills two N x N matrices with random values.
 *
 * @param test_case (struct test): The test case parameters, including N and seed.
 * @param data_out (double**): Pointer to a pointer where the generated matrix data will be stored.
 * @param N_dim (int): The dimension N of the matrices.
 * @return (int): 0 on success, non-zero on error.
 */
int generate_data(struct test, double **, int);

/**
 * @brief Reads test case parameters from an input file.
 *
 * Parses a file to extract information about test cases, such as seed, N, and output file.
 *
 * @param file_name (char*): Path to the input file.
 * @param num_tests (int*): Pointer to an integer where the number of tests found will be stored.
 * @param tests_array (struct test**): Pointer to an array of `struct test` where parsed test cases will be stored.
 * @return (int): 0 on success, non-zero on error.
 */
int read_input_file(char *, int *, struct test **);
