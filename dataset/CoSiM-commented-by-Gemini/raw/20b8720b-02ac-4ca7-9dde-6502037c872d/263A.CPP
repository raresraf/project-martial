/**
 * @file 263A.CPP
 * @brief Solution for Codeforces Problem 263A, "Beautiful Matrix".
 * @details The problem requires finding the minimum number of moves to bring the single '1' in a 5x5 matrix to the center (3,3). The number of moves is equivalent to the Manhattan distance from the '1's position to the center.
 *
 * @algorithm Direct Calculation
 * @description The program iterates through a 5x5 matrix to find the location of the number 1. Once found, it calculates the Manhattan distance from its current coordinates (i, j) to the center (2, 2) using the formula |i - 2| + |j - 2|.
 * @complexity
 * - Time: O(1), since the matrix size is constant (5x5 = 25 elements).
 * - Space: O(1), as it uses a fixed number of variables.
 */
#include <cmath>
#include <iostream>
#include <string>
using namespace std;

int main() {
  int a, b, n, total = 0;
  
  /**
   * @brief Iterates through the 5x5 matrix to locate the '1'.
   * The outer loop represents the row index (i) and the inner loop represents the column index (j).
   */
  for (int i = 0; i < 5; i++) {
    for (int j = 0; j < 5; j++) {
      cin >> n;
      /**
       * @brief When the '1' is found, calculate the Manhattan distance to the center.
       * The center of a 0-indexed 5x5 matrix is at (2, 2).
       * The Manhattan distance is the sum of the absolute differences of the coordinates.
       */
      if (n == 1) {
        total = abs(i - 2) + abs(j - 2);
      }
    }
  }
  
  // Output the total number of moves required.
  cout << total;
  return 0;
}
