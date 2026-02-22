/**
 * @file 263A.CPP
 * @brief Solution to Codeforces problem 263A - Beautiful Matrix.
 * 
 * @algorithm
 * This solution finds the minimum number of moves to center a '1' in a 5x5 grid.
 * The number of moves is equivalent to the Manhattan distance from the '1's current
 * position to the center. This implementation uses 0-indexed coordinates, so the
 * center is at (2,2).
 *
 * The algorithm is as follows:
 * 1. Declare a 5x5 integer array.
 * 2. Read the matrix elements from standard input, storing them in the array.
 * 3. While reading, identify and save the 0-indexed coordinates (row `r`, column `c`) of the '1'.
 * 4. Calculate the Manhattan distance to the center (2,2) by computing `abs(r - 2) + abs(c - 2)`.
 * 5. Print the resulting total distance.
 *
 * @complexity
 * - Time Complexity: O(1), as the matrix size is fixed at 5x5. The loops run a constant 25 times.
 * - Space Complexity: O(1), as the space for the 5x5 matrix is constant (25 integers).
 */
#include <iostream>
using namespace std;
int main() {
  int a[5][5];
  int i, j, r, c;
  /**
   * @brief Reads the 5x5 matrix and finds the 0-indexed coordinates (r, c) of the '1'.
   */
  for (i = 0; i < 5; i++) {
    for (j = 0; j < 5; j++) {
      cin >> a[i][j];
      if (a[i][j] == 1) {
        r = i;
        c = j;
      }
    }
  }
  /**
   * @brief Calculates the Manhattan distance to the center (2,2).
   * This is equivalent to abs(r - 2) and abs(c - 2).
   */
  if (r >= 2)
    r = r - 2;
  else
    r = 2 - r;
  if (c >= 2)
    c = c - 2;
  else
    c = 2 - c;
  
  // Output the total number of moves.
  cout << r + c << endl;
  return 0;
}
