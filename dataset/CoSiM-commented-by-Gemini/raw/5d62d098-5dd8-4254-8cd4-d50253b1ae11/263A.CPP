/**
 * @file 263A.CPP
 * @brief Solution for the "Beautiful Matrix" problem from Codeforces.
 * @details The problem involves a 5x5 matrix containing a single '1' and 24 '0's.
 * The goal is to find the minimum number of moves (swapping adjacent rows or columns)
 * to move the '1' to the center of the matrix, which is at position (3,3) in
 * 1-based indexing or (2,2) in 0-based indexing.
 *
 * @algorithm Manhattan Distance Calculation
 * The minimum number of moves is equivalent to the Manhattan distance between the
 * current position of the '1' and the center of the matrix.
 * 1. Read the 5x5 matrix and find the row and column (i, j) of the '1'.
 * 2. The center of the matrix is at (2, 2) using 0-based indexing.
 * 3. The number of row moves is `abs(2 - i)`.
 * 4. The number of column moves is `abs(2 - j)`.
 * 5. The total minimum moves is the sum of row and column moves.
 *
 * @complexity
 * - Time: O(1), as the matrix size is fixed at 5x5. The loops run a constant 25 times.
 * - Space: O(1), for the fixed-size 5x5 matrix.
 */

#include <bits/stdc++.h> // Includes all standard C++ libraries.
using namespace std;

int main() {
  // `a` is the 5x5 matrix.
  int a[5][5];
  // `i` and `j` are loop counters.
  // `row_of_one` and `col_of_one` will store the 0-indexed coordinates of the '1'.
  // `moves` will store the final result.
  int i, j, row_of_one, col_of_one, moves;

  /**
   * @brief Read the matrix from input and find the coordinates of the '1'.
   */
  // Invariant: After each inner loop iteration, the submatrix a[0...i][0...j] has been scanned.
  for (i = 0; i < 5; i++) {
    for (j = 0; j < 5; j++) {
      cin >> a[i][j];
      // Pre-condition: Check if the current element is the '1'.
      if (a[i][j] == 1) {
        // If it is, store its 0-indexed coordinates.
        row_of_one = i;
        col_of_one = j;
      }
    }
  }

  /**
   * @brief Calculate the Manhattan distance to the center (2, 2).
   * The number of moves is the sum of the absolute differences in row and column indices.
   */
  moves = abs(2 - row_of_one) + abs(2 - col_of_one);
  
  // Print the total minimum number of moves required.
  cout << moves;

  return 0; // Indicate successful execution.
}
