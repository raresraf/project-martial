/**
 * @file 263A.CPP
 * @brief Finds the position of '1' in a 5x5 matrix and calculates its Manhattan distance to the center (3,3).
 *
 * This program reads a 5x5 matrix of unsigned integers. It identifies the
 * 1-indexed row and column of the element with value '1'. After locating
 * this element, it computes the Manhattan distance from its position to the
 * central cell of the matrix, which is (3,3).
 *
 * Algorithm: Fixed-size matrix traversal and Manhattan distance calculation.
 * Time Complexity: O(1) because the matrix size is fixed at 5x5. The program
 *                  performs a constant number of operations (25 reads and
 *                  a few arithmetic calculations).
 * Space Complexity: O(1) as only a fixed number of variables are used
 *                   regardless of input values.
 */

#include <iostream>

void answer(size_t v) {
  /**
   * @brief Prints the calculated Manhattan distance to standard output.
   * @param v The Manhattan distance to be printed.
   */
  std::cout << v << '\n';
}

int main() {
  /**
   * @brief Main function to orchestrate matrix reading, '1' finding, and distance calculation.
   *
   * Initializes row and column trackers, iterates through the 5x5 matrix
   * to find the '1', and then calls `answer` to print the result.
   */
  // @param r Stores the 1-indexed row of the '1' element.
  // @param c Stores the 1-indexed column of the '1' element.
  int r = 0, c = 0;
  /**
   * @brief Nested loops to traverse the 5x5 matrix.
   * Invariant (outer loop): `r` and `c` store the coordinates of '1' if found in rows `1` to `i-1`.
   * Invariant (inner loop): `r` and `c` store the coordinates of '1' if found in row `i`, columns `1` to `j-1`.
   * Pre-condition: The matrix is always 5x5 and contains exactly one '1'.
   */
  for (size_t i = 1; i <= 5; ++i) {
    for (size_t j = 1; j <= 5; ++j) {
      // @param x Current integer read from the matrix.
      unsigned x;
      std::cin >> x;

      /**
       * @brief Checks if the current element is '1'.
       * If '1' is found, its 1-indexed row and column are stored.
       */
      if (x == 1)
        r = i, c = j;
    }
  }

  /**
   * @brief Calculates and prints the Manhattan distance from the '1' to the center (3,3).
   * The Manhattan distance is calculated as |x1 - x2| + |y1 - y2|.
   */
  answer(abs(3 - r) + abs(3 - c));

  return 0;
}
