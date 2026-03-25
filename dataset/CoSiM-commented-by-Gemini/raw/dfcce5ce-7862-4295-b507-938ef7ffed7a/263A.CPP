#include <stdio.h>
#include <stdlib.h> // For abs()

/**
 * @file 263A.CPP
 * @brief C-style solution for the "Beautiful Matrix" problem.
 *
 * Problem Statement:
 * Given a 5x5 matrix containing 24 zeroes and a single '1', find the minimum
 * number of moves to make the matrix "beautiful". A beautiful matrix has the
 * '1' at its center. A move consists of swapping two adjacent rows or two
 * adjacent columns.
 *
 * Algorithm:
 * The problem is equivalent to finding the Manhattan distance between the '1's
 * current position and the center of the matrix. This code uses 0-based
 * indexing for the matrix, so the center is at position (2, 2).
 *
 * 1. Read the 5x5 matrix elements. While reading, find and store the
 *    0-indexed row and column of the element that is '1'.
 * 2. Calculate the number of row swaps needed: `abs(row - 2)`.
 * 3. Calculate the number of column swaps needed: `abs(col - 2)`.
 * 4. The total minimum moves is the sum of these two values.
 *
 * Time Complexity: O(1), as the matrix size is fixed (25 elements).
 * Space Complexity: O(1), as the matrix size is fixed.
 */
int main() {
  // m: a 5x5 matrix to store the input. Declared larger for safety.
  // row, col: 0-indexed coordinates of the '1'.
  // count: total number of moves required.
  int m[10][10], row, col, i, j, count = 0;
  
  // This line is redundant and has no effect.
  // row, col;

  // Block Logic: Iterate through the 5x5 matrix to find the '1'.
  for (i = 0; i < 5; i++) {
    for (j = 0; j < 5; j++) {
      scanf("%d", &m[i][j]);
      // Pre-condition: Check if the current element is the one we are looking for.
      if (m[i][j] == 1) {
        // If found, store its 0-indexed coordinates.
        row = i;
        col = j;
      }
    }
  }

  // Calculate the Manhattan distance to the center (2, 2).
  // Vertical distance (number of row swaps).
  count += abs(row - 2);
  // Horizontal distance (number of column swaps).
  count += abs(col - 2);

  printf("%d", count);
  return 0;
}
