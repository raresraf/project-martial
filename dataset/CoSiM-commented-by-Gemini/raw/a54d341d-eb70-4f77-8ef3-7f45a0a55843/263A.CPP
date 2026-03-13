/**
 * @file 263A.CPP
 * @brief Solution for CodeForces Problem 263A - "Beautiful Matrix".
 * @author Isanchez_Aguilar
 * 
 * @details
 * The problem requires finding the minimum number of moves to make a 5x5 matrix "beautiful".
 * A beautiful matrix has a single '1' located at the center (index 2,2).
 * The only allowed move is swapping adjacent rows or columns.
 * This is equivalent to calculating the Manhattan distance between the '1's current position
 * and the center of the matrix.
 *
 * @algorithm
 * 1. Read the 5x5 matrix from standard input.
 * 2. As the matrix is read, identify the row (`oneI`) and column (`oneJ`) coordinates of the '1'.
 * 3. Calculate the Manhattan distance to the center (2,2) using the formula: |oneI - 2| + |oneJ - 2|.
 * 4. Print the result.
 *
 * @time_complexity O(1), as the matrix size is fixed at 5x5 (25 elements are always processed).
 * @space_complexity O(1), for storing the 5x5 matrix.
 */
#include <bits/stdc++.h>

using namespace std;

int main(void) {
  /**
   * Functional Utility: Decouples C++ streams from standard C streams and flushes 
   * to optimize high-throughput I/O.
   */
  ios_base::sync_with_stdio(0);
  cin.tie(0);

  int oneI = 0; // Stores the row index of the '1'.
  int oneJ = 0; // Stores the column index of the '1'.
  int mat[5][5]; // The 5x5 input matrix.

  /**
   * @brief Block to read the matrix and find the position of '1'.
   * Invariant: Before each inner loop iteration, `oneI` and `oneJ` hold the coordinates
   * of the '1' if it has been found in the preceding rows.
   */
  for (int i = 0; i < 5; ++i) {
    for (int j = 0; j < 5; ++j) {
      cin >> mat[i][j];
      
      // If the current element is 1, store its coordinates.
      if (mat[i][j] == 1)
        oneI = i, oneJ = j;
    }
  }

  // Calculate and output the Manhattan distance to the center (2, 2).
  cout << abs(oneI - 2) + abs(oneJ - 2) << "
";

  return 0;
}
