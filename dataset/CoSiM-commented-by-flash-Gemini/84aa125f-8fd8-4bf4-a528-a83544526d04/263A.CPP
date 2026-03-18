/**
 * @file 263A.CPP
 * @brief This program finds the minimum number of moves to center a '1' in a 5x5 matrix.
 * Algorithm: Iterative search to locate the '1', then Manhattan distance calculation to the center.
 * Time Complexity: O(R*C) where R and C are the dimensions of the matrix (constant 5x5, so O(1)).
 * Space Complexity: O(R*C) for storing the matrix (constant 5x5, so O(1)).
 */
#include <iostream>

int main() {
  /**
   * @brief Declares a 5x5 integer matrix to store the input.
   * Pre-condition: The input will consist of 25 integers, representing a 5x5 matrix.
   */
  int arr[5][5];
  /**
   * @brief Reads the 5x5 matrix from standard input.
   * Invariant: After each iteration of the outer loop, a full row of the matrix has been populated.
   * After each iteration of the inner loop, a single element of the current row has been populated.
   */
  for (int i = 0; i < 5; i++) {
    for (int j = 0; j < 5; j++) {
      std::cin >> arr[i][j];
    }
  }

  /**
   * @brief Iterates through the 5x5 matrix to find the element with value '1'.
   * Pre-condition: The matrix 'arr' has been populated.
   * Invariant: The loops cover every cell in the matrix exactly once.
   */
  for (int i = 0; i < 5; i++) {
    for (int j = 0; j < 5; j++) {
      /**
       * @brief Checks if the current element is '1' and calculates Manhattan distance if true.
       * Logic: If 'arr[i][j]' is '1', calculate the absolute difference in row and column
       * indices from the center (2,2) and sum them to get the Manhattan distance.
       * Post-condition: If '1' is found, its Manhattan distance to the center is printed.
       */
      if (arr[i][j] == 1) {
        int diff_x = abs(i - 2);
        int diff_y = abs(j - 2);
        std::cout << diff_x + diff_y;
      }
    }
  }
}