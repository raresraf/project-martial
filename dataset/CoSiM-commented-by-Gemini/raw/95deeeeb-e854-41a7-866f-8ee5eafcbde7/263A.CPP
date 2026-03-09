/**
 * @file 263A.CPP
 * @brief Calculates the number of moves to center a '1' in a 5x5 matrix.
 *
 * This program solves the "Beautiful Matrix" problem from Codeforces. It reads a
 * 5x5 matrix containing a single '1' and 24 '0's. The goal is to find the
 * minimum number of moves (swapping adjacent rows or columns) to move the '1'
 * to the center of the matrix (position 2,2).
 *
 * @algorithm Manhattan Distance Calculation
 * @complexity
 * - Time: O(1), as the matrix size is fixed at 5x5. The loops run a constant
 *   number of times.
 * - Space: O(1), for storing the 5x5 matrix.
 */
#include <bits/stdc++.h>

using namespace std;

int main() {
  // m: the 5x5 matrix.
  // r, c: variables to store the row and column of the '1'.
  int m[5][5], r, c;

  /**
   * @brief Block Logic: Reads the 5x5 matrix from standard input.
   */
  for (int i = 0; i < 5; i++)
    for (int j = 0; j < 5; j++)
      scanf("%d", &m[i][j]);

  /**
   * @brief Block Logic: Finds the coordinates (r, c) of the '1'.
   */
  for (int i = 0; i < 5; i++) {
    for (int j = 0; j < 5; j++) {
      if (m[i][j] == 1) {
        r = i;
        c = j;
        break; // Exit the inner loop once '1' is found.
      }
    }
  }

  /**
   * @brief Functional Utility: Calculates and prints the Manhattan distance.
   * The minimum number of moves is the sum of the horizontal and vertical
   * distances from the current position (r, c) to the center (2, 2).
   */
  printf("%d\n", abs(2 - r) + abs(2 - c));
  
  return 0;
}
