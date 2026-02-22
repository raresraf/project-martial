/**
 * @file 263A.CPP
 * @brief Solution to Codeforces problem 263A - Beautiful Matrix.
 *
 * @algorithm
 * This solution finds the minimum number of moves to bring a single '1'
 * in a 5x5 matrix to the center (position 3,3). The number of moves is
 * equivalent to the Manhattan distance.
 *
 * This implementation is memory-efficient as it does not store the entire
 * matrix. It reads each value and immediately checks if it is the '1'.
 *
 * The algorithm is as follows:
 * 1. Iterate through a 5x5 grid using 1-indexed loops (from 1 to 5).
 * 2. In each iteration, read a single integer.
 * 3. If the integer is 1, save its row and column coordinates.
 * 4. After the loops complete, calculate the Manhattan distance to the center (3,3)
 *    using the formula: |3 - r| + |3 - c|.
 * 5. Print the result using a helper function.
 *
 * @complexity
 * - Time Complexity: O(1), as the matrix size is fixed at 5x5 (25 reads).
 * - Space Complexity: O(1), as it only uses a few variables for coordinates and input,
 *   without storing the matrix.
 */
#include <iostream>

/**
 * @brief A helper function to print the final answer followed by a newline.
 * @param v The unsigned integer value to be printed.
 */
void answer(size_t v) { std::cout << v << '\n'; }

int main() {
  int r = 0, c = 0;
  /**
   * @brief Iterates through the 5x5 grid to find the 1-indexed coordinates (r, c) of the '1'.
   * This approach avoids storing the entire matrix in memory.
   */
  for (size_t i = 1; i <= 5; ++i) {
    for (size_t j = 1; j <= 5; ++j) {
      unsigned x;
      std::cin >> x;

      if (x == 1)
        r = i, c = j;
    }
  }

  // Calculate the Manhattan distance to the center (3,3) and print it.
  answer(abs(3 - r) + abs(3 - c));

  return 0;
}
