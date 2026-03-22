/**
 * @file 263A.CPP
 * @brief Solves the "Beautiful Matrix" problem (Codeforces 263A).
 * @details This program calculates the minimum number of moves to bring the single
 * '1' in a 5x5 matrix to the center position (row 3, column 3). A move is
 * defined as swapping two adjacent rows or two adjacent columns.
 *
 * @section algorithm
 * Algorithm: Manhattan Distance
 * The problem is equivalent to finding the Manhattan distance between the
 * coordinates of the '1' and the center of the matrix. The Manhattan distance
 * |x1 - x2| + |y1 - y2| gives the minimum number of single-step moves
 * (up, down, left, or right) on a grid. The code iterates through the matrix
 * inputs to find the coordinates (i, j) of the '1' and then directly
 * calculates and prints the distance to the center (3, 3).
 *
 * @section complexity
 * - Time Complexity: O(1), because the matrix size is fixed at 5x5. The loops
 *   run a constant number of times.
 * - Space Complexity: O(1), as the program does not store the matrix, only
 *   the current element being read.
 */
#include <cmath>
#include <iostream>
using namespace std;

int main() {
  int a;
  /**
   * @brief Iterates through the 5x5 matrix to find the location of the '1'.
   * The loops use 1-based indexing for rows (i) and columns (j) to match
   * the problem's coordinate system.
   */
  for (int i = 1; i <= 5; i++) {
    for (int j = 1; j <= 5; j++) {
      cin >> a;
      /**
       * @brief When the '1' is found, calculate and print the result.
       * The number of moves is the Manhattan distance from the current
       * position (i, j) to the center (3, 3).
       */
      if (a == 1)
        // |3 - i| is the number of row swaps (vertical distance).
        // |3 - j| is the number of column swaps (horizontal distance).
        cout << abs(3 - i) + abs(3 - j);
    }
  }
  return 0;
}
