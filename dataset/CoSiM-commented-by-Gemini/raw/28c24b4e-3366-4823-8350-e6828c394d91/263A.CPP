/**
 * @file 263A.CPP
 * @brief Solution to the "Beautiful Matrix" problem (263A) from Codeforces.
 * @details The problem requires finding the minimum number of moves (Manhattan distance)
 * to move a '1' in a 5x5 matrix to the center.
 *
 * @author Your Name (or leave blank)
 *
 * @algorithm This solution finds the 0-indexed coordinates of the '1' and then
 * calculates its Manhattan distance to the center (2,2).
 * 1. Use 0-indexed nested loops to iterate through the 5x5 matrix.
 * 2. Read each element of the matrix.
 * 3. If an element is '1', store its 0-indexed row and column coordinates.
 * 4. After finding the '1', calculate the distance to the center (2,2) using the
 *    Manhattan distance formula: `abs(row - 2) + abs(col - 2)`.
 * 5. Print the result.
 *
 * @complexity
 * Time Complexity: O(1) - The loops iterate a fixed number of times (25).
 * Space Complexity: O(1) - A fixed-size 5x5 array is used.
 */
#include <bits/stdc++.h>

using namespace std;
int main() {
  int arr[5][5];
  int a, b; // Variables to store the 0-indexed coordinates of the '1'.
  /**
   * @block
   * @brief Reads the 5x5 matrix and finds the 0-indexed coordinates of the '1'.
   * @pre The input stream contains 25 integers for the matrix.
   * @post `a` and `b` will hold the row and column index of the '1'.
   */
  for (int i = 0; i < 5; i++) {
    for (int j = 0; j < 5; j++) {
      cin >> arr[i][j];
      if (arr[i][j] == 1) {
        a = i;
        b = j;
      }
    }
  }
  // The center of a 0-indexed 5x5 matrix is at (2,2).
  // Calculate the Manhattan distance to the center.
  int ans = abs(a - 2) + abs(b - 2);
  cout << ans << endl;
}