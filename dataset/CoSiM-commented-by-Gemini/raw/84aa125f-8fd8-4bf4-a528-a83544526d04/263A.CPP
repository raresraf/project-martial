/**
 * @file 263A.CPP
 * @brief Solves the "Beautiful Matrix" problem.
 *
 * @details
 * The problem requires finding the minimum number of moves to bring the single '1'
 * in a 5x5 matrix to the center at position (2, 2) (0-indexed). A move consists
 * of swapping two adjacent rows or two adjacent columns.
 *
 * Algorithm:
 * The minimum number of moves is equivalent to the Manhattan distance between the
 * current position of the '1' and the center of the matrix.
 * 1. Read the 5x5 matrix to locate the position of the '1'.
 * 2. Let the coordinates of '1' be (i, j).
 * 3. The center of the 5x5 matrix is at (2, 2).
 * 4. The Manhattan distance is calculated as |i - 2| + |j - 2|.
 * 5. This sum represents the total number of row and column swaps needed.
 *
 * Time Complexity: O(1), since the matrix size is fixed at 5x5. The loops run a constant number of times.
 * Space Complexity: O(1), as it uses a fixed-size 5x5 array.
 */
#include <iostream>

int main() {
  int arr[5][5];
  /**
   * @brief Reads the 5x5 matrix from standard input.
   * Pre-condition: Input stream contains 25 integer values (0s and one 1).
   * Post-condition: The arr[5][5] is populated with the input values.
   */
  for (int i = 0; i < 5; i++) {
    for (int j = 0; j < 5; j++) {
      std::cin >> arr[i][j];
    }
  }

  /**
   * @brief Finds the '1' in the matrix and calculates its Manhattan distance to the center.
   * Pre-condition: The matrix `arr` is populated and contains exactly one '1'.
   */
  for (int i = 0; i < 5; i++) {
    for (int j = 0; j < 5; j++) {
      if (arr[i][j] == 1) {
        // Calculate the absolute difference in rows from the center (row 2).
        int diff_x = abs(i - 2);
        // Calculate the absolute difference in columns from the center (col 2).
        int diff_y = abs(j - 2);
        // The sum of differences is the Manhattan distance, which is the minimum number of moves.
        std::cout << diff_x + diff_y;
      }
    }
  }
}