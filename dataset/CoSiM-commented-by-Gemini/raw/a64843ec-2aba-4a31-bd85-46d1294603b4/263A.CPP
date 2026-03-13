/**
 * @file 263A.CPP
 * @brief Solution for CodeForces Problem 263A - "Beautiful Matrix".
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
 * 2. As the matrix is read, identify the row (`x`) and column (`y`) coordinates of the '1'.
 * 3. Calculate the Manhattan distance to the center (2,2) using the formula: |x - 2| + |y - 2|.
 * 4. Print the result.
 *
 * @time_complexity O(1), as the matrix size is fixed at 5x5 (25 elements are always processed).
 * @space_complexity O(1), for storing the 5x5 matrix.
 */
#include <bits/stdc++.h>
using namespace std;

int main() {
  int x, y; // To store the 0-indexed row and column of the '1'.
  int ans;  // To store the final answer (Manhattan distance).
  int a[5][5]; // The 5x5 input matrix.

  /**
   * @brief Block to read the matrix and find the position of '1'.
   * Invariant: Before each inner loop iteration, `x` and `y` hold the coordinates
   * of the '1' if it has been found in the preceding rows.
   */
  for (int i = 0; i < 5; i++) {
    for (int j = 0; j < 5; j++) {
      cin >> a[i][j];
      // If the current element is 1, store its coordinates.
      if (a[i][j] == 1) {
        x = i;
        y = j;
      }
    }
  }

  // Calculate the Manhattan distance to the center (2, 2).
  ans = abs(x - 2) + abs(y - 2);
  
  // Output the result.
  cout << ans;
  
  return 0;
}
