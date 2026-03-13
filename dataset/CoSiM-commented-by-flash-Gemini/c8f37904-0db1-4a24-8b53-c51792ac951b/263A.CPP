/**
 * @file 263A.CPP
 * @brief Solves the "Beautiful Matrix" problem from Codeforces (263A).
 * @details This program finds the minimum number of moves to bring the single '1'
 * in a 5x5 matrix to the center. The solution uses 1-based indexing for the grid.
 *
 * @section Algorithm
 * The problem is solved by calculating the Manhattan distance (L1 norm) from the
 * position of the '1' to the center of the grid.
 * 1. The grid is treated as 1-indexed, from (1,1) to (5,5). The center is at (3,3).
 * 2. The program iterates through each cell of the grid, reading one integer at a time.
 * 3. When the input integer is '1', its 1-based row and column indices are stored.
 *    The matrix itself is not stored, making the approach memory-efficient.
 * 4. The Manhattan distance is calculated as `abs(3 - row) + abs(3 - col)`, which
 *    represents the minimum number of single-step moves (up, down, left, right).
 *
 * @section Complexity
 * - Time Complexity: O(1), as the input size is constant (5x5 = 25 reads).
 * - Space Complexity: O(1), as only a few variables are used to store coordinates
 *   and the input value temporarily. The declared matrix `a` is unused.
 */
#include <bits/stdc++.h>
using namespace std;

int main() {
  /**
   * @var a A 2D array declared for a 5x5 matrix (with 1-based indexing buffer),
   *        but it remains unused in this implementation.
   * @var n A temporary variable to hold the value of each cell as it is read.
   * @var x, y Variables to store the 1-based row and column of the '1'.
   */
  int a[6][6];
  int n, x, y;

  /**
   * @brief Block Logic: Iterates through a 5x5 grid using 1-based indexing.
   * It reads each cell's value and captures the coordinates of the '1' when found.
   */
  for (int i = 1; i <= 5; i++) {
    for (int j = 1; j <= 5; j++) {
      cin >> n;
      if (n) { // Condition is true if n is non-zero (i.e., 1).
        x = i;
        y = j;
      }
    }
  }

  /**
   * @brief Logic: Calculates the Manhattan distance to the center (3,3) and prints it.
   * This represents the minimum number of moves required.
   */
  cout << (abs(3 - x) + abs(3 - y)) << endl;
  return 0;
}
