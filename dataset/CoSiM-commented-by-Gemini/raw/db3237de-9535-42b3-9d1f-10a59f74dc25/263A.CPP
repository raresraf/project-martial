/**
 * @file 263A.CPP
 * @brief Solution for Codeforces Problem 263A - "Beautiful Matrix".
 * @details This program calculates the minimum number of moves required to move a single '1'
 * in a 5x5 matrix to the center cell (3,3). A move consists of swapping two
 * adjacent rows or two adjacent columns.
 *
 * Algorithm:
 * The problem is equivalent to finding the Manhattan distance between the current
 * position of the '1' and the center of the matrix. The Manhattan distance is the
 * sum of the absolute differences of their coordinates.
 * 1. Iterate through the 5x5 matrix to find the coordinates (row, col) of the '1'.
 * 2. Calculate the number of row moves needed: abs(row - 3).
 * 3. Calculate the number of column moves needed: abs(col - 3).
 * 4. The sum of these two values is the minimum total moves.
 *
 * Time Complexity: O(1), as the input size is fixed to a 5x5 matrix (25 elements).
 * Space Complexity: O(1), as only a few variables are used for storage.
 */

#include <bits/stdc++.h>

using namespace std;

int main() {
  // x, y: store the 1-based coordinates of the '1'.
  // n: temporary variable to read matrix elements.
  int x, y, n;
  
  /**
   * Block Logic: Iterates through the 5x5 matrix to find the position of the '1'.
   * The loops run a fixed number of times (5x5=25).
   */
  for (int i = 0; i < 5; i++) {
    for (int j = 0; j < 5; j++) {
      scanf("%d", &n);
      // Pre-condition: If the current element is 1, store its 1-indexed coordinates.
      if (n) {
        x = i + 1;
        y = j + 1;
      }
    }
  }
  
  // Functional Utility: Calculate and print the Manhattan distance from (x,y) to the center (3,3).
  // This represents the minimum number of swaps required.
  printf("%d
", abs(x - 3) + abs(y - 3));
  return 0;
}
