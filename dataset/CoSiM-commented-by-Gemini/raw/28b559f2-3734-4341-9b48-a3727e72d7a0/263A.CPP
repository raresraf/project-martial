/**
 * @file 263A.CPP
 * @brief Solution to the "Beautiful Matrix" problem (263A) from Codeforces.
 * @details The problem is to find the minimum number of single row/column swaps
 * (i.e., the Manhattan distance) to move the single '1' in a 5x5 matrix to the
 * center at (3,3).
 *
 * @author Your Name (or leave blank)
 *
 * @algorithm This implementation finds the '1' and calculates the Manhattan distance
 * simultaneously.
 * 1. Use 1-indexed nested loops to iterate from (1,1) to (5,5), mirroring the
 *    problem's coordinate system.
 * 2. In each iteration, read the current matrix element.
 * 3. If the element is '1', immediately calculate its Manhattan distance to the center
 *    (3,3) using the formula `abs(row - 3) + abs(col - 3)` and store it.
 * 4. Since there is only one '1', the stored answer will be the final result after the
 *    loops complete.
 * 5. Print the calculated distance.
 *
 * @complexity
 * Time Complexity: O(1) - The loops run a fixed 25 times.
 * Space Complexity: O(1) - Uses a fixed-size (though oversized) array.
 */
#include <bits/stdc++.h>
using namespace std;

int main() {
  // The array `a` is oversized, a common practice in competitive programming.
  int a[60][60], ans = 0;
  /**
   * @block
   * @brief Iterates through the 5x5 matrix to find the '1' and calculate the moves.
   * @details The loops are 1-indexed to match the problem's coordinates.
   * @post `ans` will hold the Manhattan distance from the '1' to the center (3,3).
   */
  for (int i = 1; i <= 5; i++) {
    for (int j = 1; j <= 5; j++) {
      cin >> a[i][j];
      if (a[i][j] == 1)
        // If the '1' is found, calculate the Manhattan distance to the center (3,3)
        // and store it. This will be the final answer.
        ans = abs(i - 3) + abs(j - 3);
    }
  }

  cout << ans << endl;

  // your code goes here
  return 0;
}
