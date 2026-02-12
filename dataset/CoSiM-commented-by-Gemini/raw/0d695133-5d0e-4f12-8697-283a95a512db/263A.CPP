/**
 * @file 263A.CPP
 * @brief Solves the "Beautiful Matrix" problem.
 * @details This program calculates the minimum number of moves required to bring the single '1'
 * in a 5x5 grid to the center position (2,2) (using 0-based indexing). The number of moves
 * is equivalent to the Manhattan distance. This implementation uses a two-pass approach:
 * first, it reads the entire matrix into memory, and second, it searches the matrix
 * for the '1' to find its coordinates before calculating the distance.
 *
 * @algorithm Two-pass search and manual Manhattan distance calculation.
 * @complexity
 * Time complexity: O(1), as the matrix size is fixed at 5x5. The loops run a constant number of times.
 * Space complexity: O(1), for storing the 5x5 matrix.
 */
#include <bits/stdc++.h>
using namespace std;
int main() {
  // arr: the 5x5 matrix. i, j: loop counters. i1, j1: coordinates of the '1'. sum: the final result.
  int arr[5][5], i, j, i1, j1, sum;

  /**
   * @brief First pass: Read all values into the matrix.
   */
  for (i = 0; i < 5; i++) {
    for (j = 0; j < 5; j++) {
      cin >> arr[i][j];
    }
  }

  /**
   * @brief Second pass: Find the coordinates of the '1'.
   */
  for (i = 0; i < 5; i++) {
    for (j = 0; j < 5; j++) {
      if (arr[i][j] == 1) {
        i1 = i; // Store the 0-based row index.
        j1 = j; // Store the 0-based column index.
        break; // Exit the inner loop once '1' is found.
      }
    }
  }

  // Manually calculate the absolute difference for the row distance to the center (index 2).
  if (i1 < 2) {
    i1 = 2 - i1;
  } else {
    i1 = i1 - 2;
  }
  // Manually calculate the absolute difference for the column distance to the center (index 2).
  if (j1 < 2) {
    j1 = 2 - j1;
  } else {
    j1 = j1 - 2;
  }
  
  // The sum of the row and column distances is the Manhattan distance.
  sum = i1 + j1;
  cout << sum;
  return 0;
}