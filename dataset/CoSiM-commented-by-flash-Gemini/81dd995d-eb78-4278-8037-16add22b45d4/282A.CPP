/**
 * @file 282A.CPP
 * @brief Solution to Codeforces problem 282A - Bit++.
 *        This program simulates a sequence of operations on a single integer variable `x`.
 *        Each operation can either increment or decrement `x` based on the input string.
 *        The initial value of `x` is 0. This implementation reads entire operation strings
 *        and uses direct string comparison to determine the operation.
 *
 * Algorithm: Iterative Processing with Direct Pattern Matching
 *   1. Reads an integer `n`, representing the total number of operations.
 *   2. Consumes the leftover newline character from the input buffer.
 *   3. Initializes the variable `x` to 0.
 *   4. Iterates `n` times, once for each operation:
 *      a. Reads an entire line (operation string) into `polecenie`.
 *      b. Checks if `polecenie` is "++X" or "X++". If true, increments `x`.
 *      c. Otherwise, checks if `polecenie` is "--X" or "X--". If true, decrements `x`.
 *   5. After processing all operations, prints the final value of `x`.
 *
 * Time Complexity: O(N * L), where N is the number of operations and L is the
 *                  maximum length of an operation string, due to string comparisons
 *                  within the loop.
 * Space Complexity: O(L) for storing the operation string `polecenie`.
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
  int n; // Total number of operations.
  std::cin >> n; // Reads the number of operations.
  // Consumes the leftover newline character from the buffer after `std::cin >> n;`
  // This is crucial for `std::getline` to work correctly in the subsequent loop.
  std::cin.get(); 
  int x = 0; // The variable to be modified, initialized to 0.

  /**
   * Block Logic: Iterates `n` times to process each operation.
   * Invariant: `x` holds the accumulated value after previous operations.
   * Pre-condition: `n` is a non-negative integer.
   */
  for (int i = 0; i < n; i++) {
    std::string polecenie; // String representing the current operation (e.g., "++X", "X--").
    std::getline(std::cin, polecenie); // Reads the entire line for the operation.
    /**
     * Conditional Logic: Checks for increment operations.
     * Compares the `polecenie` string with known increment patterns.
     */
    if (polecenie == "++X" || polecenie == "X++") {
      x++; // Increments x.
    } 
    /**
     * Conditional Logic: Checks for decrement operations.
     * Compares the `polecenie` string with known decrement patterns.
     */
    else if (polecenie == "--X" || polecenie == "X--") {
      x--; // Decrements x.
    }
  }

  // Prints the final accumulated value of x.
  std::cout << x;

  return 0;
}