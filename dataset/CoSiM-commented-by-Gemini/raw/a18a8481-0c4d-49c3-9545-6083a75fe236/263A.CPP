/**
 * @file 263A.CPP
 * @brief Solution to the "Beautiful Matrix" problem (Codeforces 263A).
 * @details The problem is to find the minimum number of moves (adjacent row/column
 *          swaps) required to move the number '1' to the center of a 5x5 grid.
 *
 * @algorithm The solution is to calculate the Manhattan distance from the current
 *            position of the '1' to the center of the grid. For a grid with
 *            0-based indexing, the center is at (2, 2). The Manhattan distance,
 *            given by |x1 - x2| + |y1 - y2|, represents the minimum number of
 *            single-step moves (up, down, left, or right) to get from one
 *            point to another, which is equivalent to the number of row and
 *            column swaps.
 *
 * @time_complexity O(1), because the matrix size is a constant 5x5.
 * @space_complexity O(1), as it uses a fixed amount of memory.
 */
#include <bits/stdc++.h>
using namespace std;
int main() {
  long long i, j, x, y, dst, a;
  
  /**
   * @brief Iterate through the 5x5 matrix to find the 0-indexed coordinates of '1'.
   */
  for (i = 0; i < 5; i++) {
    for (j = 0; j < 5; j++) {
      cin >> a;
      if (a == 1) {
        x = i; // Store row coordinate.
        y = j; // Store column coordinate.
      }
    }
  }
  
  /**
   * @brief Calculate the Manhattan distance to the center (2, 2).
   * This is the sum of the absolute differences of the coordinates.
   */
  dst = abs(x - 2) + abs(y - 2);
  cout << dst << endl;
  return 0;
}
