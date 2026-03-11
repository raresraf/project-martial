/**
 * @file 263A.CPP
 * @brief Solves the "Beautiful Matrix" problem.
 *
 * This program calculates the minimum number of moves required to move a single '1'
 * in a 5x5 matrix to the center position (2, 2) using 0-based indexing.
 * A move consists of swapping adjacent elements.
 *
 * Algorithm: The minimum number of moves is the Manhattan distance between the
 * current position of the '1' and the center of the matrix.
 * Manhattan Distance = |x1 - x2| + |y1 - y2|.
 * In this case, it is |row - 2| + |col - 2|.
 *
 * Time Complexity: O(1), as the matrix size is fixed at 5x5. The loops run a
 *                  constant number of times (25).
 * Space Complexity: O(1), for storing the 5x5 matrix.
 */
#include <iostream>
#include <cmath> // For std::abs

using namespace std;

int main() {
  int arr[5][5]; // The 5x5 matrix.
  int row, col;  // Variables to store the 0-indexed coordinates of the '1'.

  /**
   * @brief Read the matrix and find the coordinates of the '1'.
   *
   * Pre-condition: The input stream contains 25 integers (0 or 1), with exactly one '1'.
   * Invariant: After each inner loop iteration, one element of the matrix is read.
   *            If the element is '1', its coordinates are stored.
   */
  for (int i = 0; i < 5; i++) {
    for (int j = 0; j < 5; j++) {
      cin >> arr[i][j];
      if (arr[i][j] == 1) {
        row = i;
        col = j;
      }
    }
  }
  
  // Calculate the Manhattan distance from the current position (row, col) to the center (2, 2).
  int moves = abs(2 - row) + abs(2 - col);
  
  // Output the minimum number of moves.
  cout << moves;
  
  return 0;
}
