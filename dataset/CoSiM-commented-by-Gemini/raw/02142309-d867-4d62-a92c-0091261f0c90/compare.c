
/**
 * @file compare.c
 * @brief Utility for robust comparison of two matrices stored in binary files.
 *
 * This program reads two matrices from separate files, assuming they are stored
 * in row-major order as double-precision floating-point numbers. It then
 * performs an element-wise comparison with a specified numerical precision
 * to determine if the matrices are equivalent.
 *
 * Algorithm: Memory-mapped element-wise comparison with a tolerance.
 * Time Complexity: O(N^2) where N is the dimension of the square matrix.
 * Space Complexity: O(1) additional space beyond memory-mapped files.
 */
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/mman.
#include <unistd.h>
#include <math.h>

/**
 * @def check_err(a, b, err)
 * @brief Checks if the absolute difference between two floating-point numbers
 * is within a given tolerance.
 * @param a The first number.
 * @param b The second number.
 * @param err The tolerance.
 * @return 0 if the numbers are within the tolerance, -1 otherwise.
 */
#define check_err(a,b,err) ((fabs((a) - (b)) <= (err)) ? 0 : -1)


/**
 * @brief Compares two matrix files for equality within a given precision.
 * @param file_path1 Path to the first matrix file.
 * @param file_path2 Path to the second matrix file.
 * @param precision The maximum allowed difference between corresponding elements.
 * @return 0 if the matrices are equal, -1 otherwise.
 *
 * This function uses memory mapping for efficient file access. It first checks
 * if the file sizes are equal and then proceeds to compare the matrices element
 * by element.
 */
int cmp_files(char const *file_path1, char const *file_path2, double precision) {
	struct stat fileInfo1, fileInfo2;
	double *mat1, *mat2;
	int i, j, fd1, fd2, N, ret = 0;

	// Functional Utility: Opens file descriptors for both matrix files.
	fd1 = open(file_path1, O_RDONLY, (mode_t)0600);
	fd2 = open(file_path2, O_RDONLY, (mode_t)0600);

	// Functional Utility: Retrieves file metadata to obtain their sizes.
	fstat(fd1, &fileInfo1);
	fstat(fd2, &fileInfo2);

	/**
	 * Block Logic: Compares file sizes to ensure preliminary structural compatibility.
	 * Pre-condition: Both file descriptors are valid.
	 * Invariant: If file sizes differ, matrices cannot be identical.
	 */
	if(fileInfo1.st_size != fileInfo2.st_size) {
		printf("Files length differ\n");
		close(fd1);
		close(fd2);
		return -1;
	}

	/**
	 * Block Logic: Memory-maps the first file into the process's address space for efficient access.
	 * Pre-condition: Valid file descriptor and size.
	 * Invariant: `mat1` points to the start of the memory-mapped content, or `MAP_FAILED` on error.
	 */
	mat1 = (double*) mmap(0, fileInfo1.st_size, PROT_READ, MAP_SHARED, fd1, 0);
	if (mat1 == MAP_FAILED)
	{
		close(fd1);
		close(fd2);
		printf("Error mmapping the first file");
		return -1;
	}

	/**
	 * Block Logic: Memory-maps the second file into the process's address space.
	 * Pre-condition: Valid file descriptor and size; first file successfully mapped.
	 * Invariant: `mat2` points to the start of the memory-mapped content, or `MAP_FAILED` on error.
	 */
	mat2 = (double*) mmap(0, fileInfo2.st_size, PROT_READ, MAP_SHARED, fd2, 0);
	if (mat2 == MAP_FAILED)
	{
		munmap(mat1, fileInfo1.st_size);
		close(fd1);
		close(fd2);
		printf("Error mmapping the second file");
		return -1;
	}

	// Functional Utility: Computes the dimension N of the square matrix from the file size.
	N = sqrt(fileInfo1.st_size / sizeof(double));

	/**
	 * Block Logic: Iterates through both matrices element by element to perform a comparison
	 * with the specified precision.
	 * Pre-condition: Both matrices are successfully memory-mapped and have identical dimensions.
	 * Invariant: `ret` indicates whether a difference exceeding `precision` has been found.
	 */
	for (i = 0; i < N; i++ ) {
		for (j = 0; j< N; j++) {
			// Inline: Compares matrix elements at [i, j] for equality within the given precision.
			ret = check_err(mat1[i * N + j], mat2[i * N + j], precision); 
			if (ret != 0) {
				printf("Matrixes differ on index [%d, %d]. Expected %.8lf got %.8lf\n",
					i, j, mat1[i * N + j], mat2[i * N + j]);
				// Functional Utility: Jumps to cleanup to release resources if a mismatch is found.
				goto done;
			}
		}
	}

done:
	// Functional Utility: Unmaps the memory regions to release virtual memory resources.
	munmap(mat1, fileInfo1.st_size);
	munmap(mat2, fileInfo2.st_size);

	// Functional Utility: Closes the file descriptors.
	close(fd1);
	close(fd2);

	// Functional Utility: Returns the comparison result (0 for equal, -1 for different).
	return ret;
}
/**
 * @brief Main entry point of the program.
 * @param argc Number of command-line arguments.
 * @param argv Array of command-line arguments.
 * @return 0 on success, -1 on failure.
 *
 * The program expects three command-line arguments: the paths to the two
 * matrix files and the tolerance for the comparison.
 */
int main(int argc, const char **argv)
{    
	double precision;
	int ret = 0;

	/**
	 * Block Logic: Validates the number of command-line arguments.
	 * Invariant: Requires exactly three arguments: two file paths and a precision value.
	 */
	if(argc < 4) {
		printf("Usage: %s mat1 mat2 tolerance\n",argv[0]);
		exit(-1);
	}

	// Functional Utility: Parses the precision value from the third command-line argument.
	sscanf(argv[3], "%lf", &precision);

	// Functional Utility: Invokes the file comparison logic with the provided inputs.
	ret = cmp_files(argv[1],argv[2],precision);
	
	// Functional Utility: Prints the comparison result to standard output.
	printf("%s %s %s %s\n", argv[0], argv[1], argv[2], (ret == 0 ? "OK" : "Incorrect results!"));
	return ret;
}
