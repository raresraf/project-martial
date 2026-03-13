/**
 * @file 263A.CPP
 * @brief Solves the "Beautiful Matrix" problem from Codeforces.
 * @details This program calculates the minimum number of moves required to move a '1'
 * to the center of a 5x5 matrix. The allowed moves are single steps up, down, left, or right.
 *
 * @section Algorithm
 * The problem is equivalent to finding the Manhattan distance (or L1 norm) between
 * the coordinates of the '1' and the center of the grid, which is at position (2, 2)
 * in a 0-indexed 5x5 matrix.
 * 1. A 5x5 integer matrix is read from standard input.
 * 2. During input, the row and column indices of the element equal to '1' are stored.
 * 3. The Manhattan distance is calculated as `abs(row_of_1 - 2) + abs(col_of_1 - 2)`.
 *
 * @section Complexity
 * - Time Complexity: O(1), as the matrix size is constant (5x5 = 25 elements).
 * - Space Complexity: O(1), as the storage is a fixed-size 5x5 matrix.
 */

#include <cmath>
#include <iostream>

using namespace std;

int main() {
  /**
   * @var i, j Loop counters.
   * @var m Stores the final Manhattan distance (number of moves).
   * @var pr, pc Stores the row and column position of the '1'.
   * @var a The 5x5 integer matrix.
   */
  int i, j, m = 0, pr, pc;
  int a[5][5];

  /**
   * @brief Block Logic: Reads the 5x5 matrix from standard input.
   * @invariant During the iteration, if the current element `a[i][j]` is '1',
   * its coordinates are captured in `pr` and `pc`.
   */
  for (i = 0; i < 5; i++) {
    for (j = 0; j < 5; j++) {
      cin >> a[i][j];
      if (a[i][j] == 1) {
        pr = i;
        pc = j;
      }
    }
  }

  /**
   * @brief Logic: Calculates the Manhattan distance from the element '1' to the center (2, 2).
   * This distance represents the minimum number of single-step moves required.
   */
  m = abs(pr - 2) + abs(pc - 2);
  
  // Output the final calculated number of moves.
  cout << m;
  return 0;
}