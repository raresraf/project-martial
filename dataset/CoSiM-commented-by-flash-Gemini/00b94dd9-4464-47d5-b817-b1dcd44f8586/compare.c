
/**
 * @file compare.c
 * @brief This utility compares two binary files, assumed to contain square
 *        matrices of double-precision floating-point values. It uses memory-mapped
 *        I/O for efficient access and performs an element-by-element comparison
 *        within a specified numerical precision (tolerance).
 *
 * This tool is typically used to verify the correctness of matrix computation
 * results by comparing an output file against a known-good reference file.
 */

#include <stdlib.h>  // For exit(), malloc(), free()
#include <stdio.h>   // For printf(), fprintf(), fopen(), fclose(), sscanf()
#include <stdint.h>  // For standard integer types (e.g., uint8_t)
#include <fcntl.h>   // For open()
#include <sys/stat.h> // For fstat()
#include <sys/mman.h> // For mmap(), munmap()
#include <unistd.h>  // For close()
#include <math.h>    // For fabs()

/**
 * @brief Macro to check if two double-precision floating-point numbers are approximately equal.
 *
 * This macro compares the absolute difference between two numbers (`a` and `b`)
 * against a given error tolerance (`err`).
 *
 * @param a The first double value.
 * @param b The second double value.
 * @param err The maximum allowed absolute difference (tolerance).
 * @return 0 if the absolute difference is less than or equal to `err`, indicating
 *         the numbers are considered equal; otherwise, returns -1.
 * Algorithm: Absolute difference comparison.
 */
#define check_err(a,b,err) ((fabs((a) - (b)) <= (err)) ? 0 : -1)


/**
 * @brief Compares two binary files containing matrices of double-precision floating-point values.
 *
 * This function memory-maps the two input files and then performs an element-by-element
 * comparison. It assumes both files contain a square matrix of `double` values.
 * The comparison considers values equal if their absolute difference is within
 * the specified `precision`.
 *
 * @param file_path1 The path to the first binary file.
 * @param file_path2 The path to the second binary file.
 * @param precision The maximum allowed absolute difference between corresponding
 *                  elements for them to be considered equal.
 * @return 0 if the files contain identical matrices within the given precision,
 *         -1 if they differ or if an error occurs during file operations or memory mapping.
 */
int cmp_files(char const *file_path1, char const *file_path2, double precision) {
	struct stat fileInfo1, fileInfo2; // Structures to store file metadata.
	double *mat1, *mat2;             // Pointers to the memory-mapped matrix data.
	int i, j, fd1, fd2, N;           // Loop counters, file descriptors, matrix dimension.
	int ret = 0;                     // Return value, initialized to success (0).

	// Open the first and second binary files for reading.
	fd1 = open(file_path1, O_RDONLY, (mode_t)0600);
	fd2 = open(file_path2, O_RDONLY, (mode_t)0600);

	// Retrieve file status (metadata) for both files.
	fstat(fd1, &fileInfo1);
	fstat(fd2, &fileInfo2);

	/**
	 * Block Logic: Checks if the sizes of the two files are identical.
	 * Functional Utility: Files containing matrices of different sizes cannot be compared.
	 */
	if(fileInfo1.st_size != fileInfo2.st_size) {
		printf("Files length differ\n");
		close(fd1);
		close(fd2);
		return -1; // Return error if file sizes are different.
	}

	/**
	 * Block Logic: Memory-maps the first file into the process's address space.
	 * Functional Utility: Provides direct pointer access to file content, improving I/O performance.
	 * Error Handling: Checks if mmap operation was successful.
	 */
	mat1 = (double*) mmap(0, fileInfo1.st_size, PROT_READ, MAP_SHARED, fd1, 0);
	if (mat1 == MAP_FAILED)
	{
		close(fd1);
		close(fd2);
		printf("Error mmapping the first file");
		return -1; // Return error on mmap failure.
	}

	/**
	 * Block Logic: Memory-maps the second file into the process's address space.
	 * Functional Utility: Provides direct pointer access to file content, improving I/O performance.
	 * Error Handling: Checks if mmap operation was successful and cleans up previous mmap if it fails.
	 */
	mat2 = (double*) mmap(0, fileInfo2.st_size, PROT_READ, MAP_SHARED, fd2, 0);
	if (mat2 == MAP_FAILED)
	{
		munmap(mat1, fileInfo1.st_size); // Clean up the first memory map.
		close(fd1);
		close(fd2);
		printf("Error mmapping the second file");
		return -1; // Return error on mmap failure.
	}

	// Calculate the dimension N of the square matrix.
	// Assumes the file size is N*N*sizeof(double).
	N = sqrt(fileInfo1.st_size / sizeof(double));

	/**
	 * Block Logic: Iterates through each element of the memory-mapped matrices and compares them.
	 * Algorithm: Element-wise comparison using nested loops.
	 * Invariant: If `check_err` returns non-zero, a difference is found, and the comparison stops.
	 */
	for (i = 0; i < N; i++ ) {
		for (j = 0; j< N; j++) {
			ret = check_err(mat1[i * N + j], mat2[i * N + j], precision);
			if (ret != 0) {
				printf("Matrixes differ on index [%d, %d]. Expected %.8lf got %.8lf\n",
						i, j, mat1[i * N + j], mat2[i * N + j]);
				goto done; // Jump to cleanup on first difference.
			}
		}
	}

done:
	/**
	 * Block Logic: Cleans up memory maps and closes file descriptors.
	 * Functional Utility: Releases system resources acquired during file processing.
	 */
	munmap(mat1, fileInfo1.st_size); // Unmap the first file.
	munmap(mat2, fileInfo2.st_size); // Unmap the second file.

	close(fd1); // Close the first file descriptor.
	close(fd2); // Close the second file descriptor.

	return ret; // Return the comparison result.
}

/**
 * @brief Main entry point for the matrix comparison utility.
 *
 * Parses command-line arguments: two file paths for the matrices and a
 * precision (tolerance) for floating-point comparison. It then calls
 * `cmp_files` to perform the comparison and prints a summary result.
 *
 * @param argc The number of command-line arguments.
 * @param argv An array of strings representing the command-line arguments.
 *             Expected format: `<executable_name> <matrix_file1> <matrix_file2> <tolerance>`.
 * @return 0 if the matrices are identical within the given precision,
 *         a non-zero value if they differ or if invalid arguments are provided.
 */
int main(int argc, const char **argv)
{
	double precision; // Stores the precision (tolerance) for comparison.
	int ret = 0;      // Stores the return value of the comparison function.

	/**
	 * Block Logic: Validates the number of command-line arguments.
	 * Error Handling: Prints usage message and exits if arguments are insufficient.
	 */
	if(argc < 4) {
		printf("Usage: %s mat1 mat2 tolerance\n",argv[0]);
		exit(-1); // Exit with error code.
	}

	// Parse the precision (tolerance) from the third command-line argument.
	sscanf(argv[3], "%lf", &precision);

	// Call the `cmp_files` function to perform the actual matrix comparison.
	ret = cmp_files(argv[1],argv[2],precision);
	
	// Print a summary of the comparison result.
	printf("%s %s %s %s\n", argv[0], argv[1], argv[2], (ret == 0 ? "OK" : "Incorrect results!"));
	return ret; // Return the overall success/failure of the comparison.
}
