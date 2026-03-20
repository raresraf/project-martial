/**
 * @file 263A.CPP
 * @brief Solution for Codeforces problem 263A, "Beautiful Matrix".
 * @author Nikhil Nagrale (nikhilnagrale2 on EveryPlatform)
 * @details This program calculates the minimum number of moves (Manhattan distance)
 * required to bring a single '1' in a 5x5 grid to the center. This implementation
 * uses 1-based indexing for grid traversal.
 *
 * @section Algorithm
 * The problem is solved by finding the Manhattan distance from the coordinates of the
 * '1' to the center of the grid. With 1-based indexing, the center of a 5x5 grid is at (3, 3).
 *
 * 1. The code uses nested loops to iterate from 1 to 5 for both rows and columns.
 * 2. It reads each grid element into a temporary variable `n`.
 * 3. When an element with value 1 is found (`if (n)` evaluates to true), its
 *    1-based coordinates (`i` and `j`) are stored in `x` and `y`.
 * 4. The Manhattan distance is then calculated as `abs(3 - x) + abs(3 - y)`.
 * 5. The result is printed.
 *
 * Note: A 6x6 array `a` is declared, likely to facilitate 1-based indexing without
 * going out of bounds, but it is never actually written to or read from, making it redundant.
 *
 * @subsection Complexity
 * - Time Complexity: O(1), because the loops run a fixed number of times (25).
 * - Space Complexity: O(1), as the memory usage is constant.
 */
// code by Nikhil Nagrale
// nikhilnagrale2 on EveryPlatform
#include <bits/stdc++.h>
using namespace std;

int main() {
  // `a` is declared to accommodate 1-based indexing but is not used in the code.
  int a[6][6];
  // n: holds the value of each element read from the grid.
  // x, y: store the 1-based row and column of the '1'.
  int n, x, y;
  /**
   * @brief This nested loop reads the 5x5 grid using 1-based indexing
   * to find the location of the number 1.
   */
  for (int i = 1; i <= 5; i++) {
    for (int j = 1; j <= 5; j++) {
      cin >> n;
      // `if (n)` is a concise way to test `if (n == 1)`.
      if (n) {
        x = i;
        y = j;
      }
    }
  }
  /**
   * @brief Calculates and prints the Manhattan distance to the center (3, 3).
   * This distance represents the minimum number of single-step moves (up, down,
   * left, or right) to move the '1' to the center of the grid.
   */
  cout << (abs(3 - x) + abs(3 - y)) << endl;
  return 0;
}