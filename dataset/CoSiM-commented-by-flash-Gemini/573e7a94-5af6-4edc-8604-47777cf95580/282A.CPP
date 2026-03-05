/**
 * @file 282A.CPP
 * @brief Solution to a programming problem that simulates a sequence of
 *        increment and decrement operations on a single variable 'x'.
 *        The operations can be "X++", "++X", "X--", or "--X".
 *        The program calculates the final value of 'x' starting from 0.
 *
 * Algorithm: Iterative processing of operations.
 *            The algorithm initializes a variable `x` to 0. It then reads
 *            a specified number of operations. For each operation string,
 *            it checks if it contains "++" (increment) or "--" (decrement)
 *            and updates `x` accordingly.
 * Time Complexity: O(N * L), where N is the number of operations and L is
 *                  the length of an operation string. In this problem, L is a
 *                  small constant (3 characters), so it's effectively O(N).
 * Space Complexity: O(L) for storing the current operation string.
 */

#include <iostream>
using namespace std;

int main() {
  // Functional Utility: Initializes `n` for the number of operations,
  // `x` as the variable to be modified (starting at 0), and `ans` (unused in this code).
  int n, x = 0, ans = 0;
  // Block Logic: Reads the total number of operations (n) from standard input.
  cin >> n;
  string o; // Declares a string to store each operation.
  /**
   * Block Logic: Iterates `n` times to read and process each operation.
   * Invariant: In each iteration, `i` is the current operation number, and `x`
   *            holds the cumulative result of operations processed so far.
   */
  for (int i = 0; i < n; i++) {
    // Block Logic: Reads the current operation string into `o`.
    cin >> o;
    // Pre-condition: Checks if the operation string signifies an increment.
    // Functional Utility: Compares the operation string `o` with "X++" or "++X".
    if (o == "X++" || o == "++X")
      // Block Logic: Increments `x` if the operation is an increment.
      x++;
    // Pre-condition: Checks if the operation string signifies a decrement.
    else if (o == "X--" || o == "--X")
      // Block Logic: Decrements `x` if the operation is a decrement.
      x--;
  }
  // Post-condition: Prints the final calculated value of `x`.
  cout << x;
  return 0;
}