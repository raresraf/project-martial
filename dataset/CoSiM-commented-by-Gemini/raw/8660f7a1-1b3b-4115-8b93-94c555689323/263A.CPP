/**
 * @file 263A.CPP
 * @brief Solves the "Beautiful Matrix" problem using 1-based indexing.
 *
 * @details
 * The program finds the minimum number of moves (swapping adjacent rows or
 * columns) to move the single '1' in a 5x5 matrix to the center. The minimum
 * number of moves is the Manhattan distance. This implementation uses 1-based
 * indexing for matrix coordinates.
 *
 * Algorithm:
 * 1. Iterate through a 5x5 grid using 1-based indices (from 1 to 5).
 * 2. While reading the input, check for the position of the '1'.
 * 3. Store the 1-indexed coordinates (x, y) of the '1'.
 * 4. The center of the matrix in this indexing scheme is (3, 3).
 * 5. Calculate the Manhattan distance: `abs(3 - x) + abs(3 - y)`.
 * 6. Print the result.
 *
 * Time Complexity: O(1), as the matrix size is fixed at 5x5.
 * Space Complexity: O(1), as it only uses a few variables to store coordinates and input.
 */
#include <bits/stdc++.h>

using namespace std;

int main() {

  int x, y; // To store the 1-based coordinates of the '1'.
  int in;   // Temporary variable for input.

  /**
   * @brief Reads the matrix input and finds the 1-indexed position of the '1'.
   */
  for (int i = 1; i <= 5; i++) {
    for (int j = 1; j <= 5; j++) {
      cin >> in;
      if (in) { // This is true when the input `in` is 1.
        x = i;
        y = j;
      }
    }
  }

  // The center of the 5x5 matrix is at (3, 3) with 1-based indexing.
  // Calculate the Manhattan distance from the '1's position (x, y) to the center.
  cout << abs(3 - x) + abs(3 - y) << endl;

  return 0;
}