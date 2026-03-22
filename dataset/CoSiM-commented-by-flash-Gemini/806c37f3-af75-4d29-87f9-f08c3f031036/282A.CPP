/**
 * @file 282A.CPP
 * @brief Solution to Codeforces problem 282A - Bit++.
 *        This program simulates a sequence of operations on a single integer variable `X`.
 *        Each operation can either increment or decrement `X` based on the input string.
 *        The initial value of `X` is 0. This implementation identifies the operation
 *        by searching for '+' or '-' characters within the operation string.
 *
 * Algorithm:
 *   1. Reads an integer `n` (referred to as `x` in the code), representing the total
 *      number of operations.
 *   2. Initializes the variable `k` (representing `X` from the problem) to 0.
 *   3. Iterates `n` times, once for each operation:
 *      a. Reads an operation string `s`.
 *      b. Iterates through the characters of `s`.
 *      c. If a '+' character is found, `k` is incremented, and the inner loop breaks.
 *      d. If a '-' character is found, `k` is decremented, and the inner loop breaks.
 *   4. After processing all operations, prints the final value of `k`.
 *
 * Time Complexity: O(N * L_avg), where N is the number of operations and L_avg is the
 *                  average length of an operation string, as the inner loop breaks
 *                  once a '+' or '-' is found. In the worst case, it can be O(N * L_max).
 * Space Complexity: O(L_max) for storing the operation string `s`.
 */

#include <bits/stdc++.h> // Includes common C++ standard library headers.
using namespace std;

int main() {
  /**
   * @brief Main function that solves the "Bit++" problem.
   *
   * Reads a series of increment/decrement operations and applies them to an integer variable,
   * then prints the final value of the variable.
   *
   * @return int Returns 0 upon successful execution.
   */
  int n_operations; // Variable `x` in the original code, representing the number of operations.
  cin >> n_operations;

  int result_x = 0; // Variable `k` in the original code, representing the final value of X.

  /**
   * Block Logic: Iterates through each of the `n_operations`.
   * Invariant: `result_x` holds the accumulated value after previous operations.
   * Pre-condition: `n_operations` is a non-negative integer.
   */
  for (int i = 0; i < n_operations; i++) {
    string s; // String representing the current operation (e.g., "++X", "X--").
    cin >> s;
    /**
     * Block Logic: Iterates through the characters of the operation string to find '+' or '-'.
     * The loop breaks immediately once an operation character is found, as only one
     * increment/decrement occurs per string.
     */
    for (int j = 0; j < s.length(); j++) {
      /**
       * Conditional Logic: Checks for the '+' character to perform an increment.
       * Inline: If '+' is found, increments `result_x` and breaks the inner loop.
       */
      if (s[j] == '+') {
        result_x++;
        break; // Only one operation per string, so break after finding it.
      }
      /**
       * Conditional Logic: Checks for the '-' character to perform a decrement.
       * Inline: If '-' is found, decrements `result_x` and breaks the inner loop.
       */
      if (s[j] == '-') {
        result_x--;
        break; // Only one operation per string, so break after finding it.
      }
    }
  }

  // Prints the final accumulated value of `result_x`.
  cout << result_x;
  return 0;
}
