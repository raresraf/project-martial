/**
 * @file 282A.CPP
 * @brief Solution to Codeforces problem 282A - Bit++.
 *        This program simulates a sequence of operations on a single integer variable `x`.
 *        Each operation can either increment or decrement `x` based on the input string.
 *        The initial value of `x` is 0.
 *
 * Algorithm:
 *   1. Reads an integer `n`, representing the total number of operations.
 *   2. Initializes the variable `x` to 0.
 *   3. Iterates `n` times, once for each operation:
 *      a. Reads an operation string `s`.
 *      b. Checks if `s` contains "++". If it does, increments `x`.
 *      c. Checks if `s` contains "--". If it does, decrements `x`.
 *   4. After processing all operations, prints the final value of `x`.
 *
 * Time Complexity: O(N * L), where N is the number of operations and L is the
 *                  maximum length of an operation string, due to the `string::find`
 *                  operation within the loop.
 * Space Complexity: O(L) for storing the operation string `s`.
 */

#include <iostream>
#include <string>

int main() {
  /**
   * @brief Main function that solves the "Bit++" problem.
   *
   * Reads a series of increment/decrement operations and applies them to an integer variable,
   * then prints the final value of the variable.
   *
   * @return int Returns 0 upon successful execution.
   */
  int n;     // Number of operations.
  int x = 0; // The variable to be modified, initialized to 0.
  std::cin >> n; // Reads the total number of operations.
  /**
   * Block Logic: Iterates `n` times to process each operation.
   * Invariant: `x` holds the accumulated value after previous operations.
   * Pre-condition: `n` is a non-negative integer.
   */
  for (int i = 0; i < n; i++) {
    std::string s; // String representing the current operation (e.g., "++X", "X--").
    std::cin >> s;
    /**
     * Conditional Logic: Checks for the presence of "++" substring to determine increment operation.
     * `std::string::npos` indicates "not found".
     */
    if (s.find("++") != std::string::npos) {
      x += 1; // Increments x.
    }
    /**
     * Conditional Logic: Checks for the presence of "--" substring to determine decrement operation.
     * It's important to note that a single string won't contain both "++" and "--".
     */
    if (s.find("--") != std::string::npos) {
      x -= 1; // Decrements x.
    }
  }
  // Prints the final accumulated value of x.
  std::cout << x << std::endl;
}