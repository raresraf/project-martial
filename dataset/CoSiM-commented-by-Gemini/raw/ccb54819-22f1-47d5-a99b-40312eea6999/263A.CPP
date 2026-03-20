/**
 * @file 263A.CPP
 * @brief Solution for the "Beautiful Matrix" problem.
 * @details The problem involves a 5x5 matrix containing a single '1' and twenty-four '0's.
 * The goal is to find the minimum number of single-step moves (swapping with an
 * adjacent element) required to move the '1' to the center of the matrix, at position (3, 3).
 *
 * Algorithm:
 * The minimum number of moves is equal to the Manhattan distance between the '1's
 * current position (i, j) and the target center position (3, 3). The Manhattan
 * distance is calculated as |i - 3| + |j - 3|.
 *
 * The code iterates through the 5x5 grid, reading each element. When it finds the
 * '1', it calculates and prints this distance and then effectively finishes.
 *
 * Time Complexity: O(1), as the matrix size is a constant 5x5.
 * Space Complexity: O(1).
 */
#include <cmath>   // For std::abs
#include <iostream>

using namespace std;

int main() {
  int x = 0; // Variable to store the value of each matrix element.

  /**
   * Block Logic: Iterate through the 5x5 grid to find the '1'.
   * The loops use 1-based indexing for rows (i) and columns (j) from 1 to 5.
   */
  for (int i = 1; i <= 5; ++i) {
    for (int j = 1; j <= 5; ++j) {
      cin >> x;
      // Pre-condition: 'x' is the element at position (i, j).
      if (x == 1) {
        // If the element is '1', calculate the Manhattan distance to the center (3, 3).
        // This distance is the minimum number of moves required.
        cout << abs(i - 3) + abs(j - 3) << endl;
        // The program could exit here, but continuing the loops is harmless
        // as the 'if' condition will not be met again.
      }
    }
  }
  return 0;
}
