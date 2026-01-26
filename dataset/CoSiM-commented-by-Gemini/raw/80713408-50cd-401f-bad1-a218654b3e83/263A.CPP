/**
 * @file 263A.CPP
 * @brief Solution to a competitive programming problem involving a 5x5 matrix.
 * @details This program reads a 5x5 matrix containing a single '1' and calculates
 * the Manhattan distance required to move the '1' to the center of the matrix (position 2,2).
 * The Manhattan distance is the sum of the absolute differences of the coordinates.
 *
 * @algorithm The algorithm finds the coordinates of the '1' in the matrix and then
 * calculates the Manhattan distance to the center.
 * @complexity
 * Time complexity: O(1), as the matrix size is fixed at 5x5.
 * Space complexity: O(1), as it uses a constant amount of extra space.
 */
#include <cmath>
#include <iostream>
using namespace std;

/**
 * @brief Main function to execute the solution.
 */
int main() {
  /**
   * @param temp A temporary variable to store the matrix element being read.
   * @param pos A pair to store the coordinates (row, column) of the '1'.
   */
  int temp;
  pair<int, int> pos;

  /**
   * @brief This nested loop reads the 5x5 matrix and finds the position of the '1'.
   */
  for (int i = 0; i < 5; i++) {
    for (int j = 0; j < 5; j++) {
      cin >> temp;
      if (temp == 1) {
        pos.first = i;
        pos.second = j;
      }
    }
  }

  /**
   * @brief Calculate the Manhattan distance to the center (2,2).
   * The center of a 0-indexed 5x5 matrix is at (2,2).
   */
  int res = abs(2 - pos.first) + abs(2 - pos.second);

  cout << res << endl;

  return 0;
}
