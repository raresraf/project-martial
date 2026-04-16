
/**
 * @file utils.h
 * @brief Utility functions and type definitions for a matrix solver test framework.
 * @details This header defines the core components for testing different matrix solver
 * implementations. It specifies the function signature for a solver, a struct
 * for test configuration, and declares functions for test execution, data
 * generation, and file I/O.
 */
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

/**
 * @brief Function pointer type for a matrix solver.
 * @param N The dimension of the square matrices.
 * @param A A pointer to the first input matrix (N x N).
 * @param B A pointer to the second input matrix (N x N).
 * @return A pointer to the resulting matrix (N x N).
 */
typedef double* (*Solver)(int, double *, double*);

/**
 * @brief Generates a random double within a specified limit.
 * @param limit The upper bound for the random number generation.
 * @return A random double value in the range [-limit, limit].
 */
#define get_rand_double(limit) ((((double)rand()) / RAND_MAX) * (2 * limit) - limit)

/**
 * @brief A struct to hold test configuration parameters.
 */
struct test {
	int seed;  /**< The seed for the random number generator. */
	int N;     /**< The dimension of the square matrices. */
	
	char output_save_file[100]; /**< The path to the file where the output will be saved. */
};

/**
 * @brief The function to be implemented by different solver versions.
 * @param N The dimension of the square matrices.
 * @param A A pointer to the first input matrix.
 * @param B A pointer to the second input matrix.
 * @return A pointer to the result matrix.
 */
double* my_solver(int, double *, double *);

/**
 * @brief Runs a test case for a given solver.
 * @param t The test configuration.
 * @param solver The solver implementation to test.
 * @param time A pointer to a float where the execution time will be stored.
 * @return An integer indicating the test result (0 for success).
 */
int run_test(struct test, Solver, float *);

/**
 * @brief Frees the memory allocated for the matrices.
 * @param data A pointer to an array of matrix pointers.
 */
void free_data(double **);

/**
 * @brief Generates random matrix data for a test.
 * @param t The test configuration.
 * @param data A pointer to an array of matrix pointers to be filled.
 * @param N The dimension of the matrices.
 * @return An integer indicating the result (0 for success).
 */
int generate_data(struct test, double **, int);

/**
 * @brief Reads the test configuration from an input file.
 * @param file_path The path to the input file.
 * @param num_tests A pointer to an integer where the number of tests will be stored.
 * @param tests A pointer to an array of test structs to be filled.
 * @return An integer indicating the result (0 for success).
 */
int read_input_file(char *, int *, struct test **);
