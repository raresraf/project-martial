/**
 * @file 282A.CPP
 * @brief Solution for Codeforces Problem 282A - "Bit++".
 *
 * This program simulates a basic programming language with a single variable 'X'
 * initialized to 0. It processes a series of operations, either incrementing
 * or decrementing 'X'. The operations can be in either prefix or postfix form
 * (e.g., "++X" or "X++"). The program calculates and prints the final value of 'X'.
 *
 * @algorithm Simple loop-based simulation.
 * @time_complexity O(N), where N is the number of operations. The program
 *                  reads and processes each operation in constant time within a loop.
 * @space_complexity O(1), as it only stores the result and the current operation string.
 */
#include "string"
#include <iostream>
using namespace std;

/**
 * @brief Main function to simulate the operations.
 */
int main(int argc, const char *argv[]) {
  int n;
  int res = 0;
  // Read the number of operations.
  cin >> n;

  /**
   * @brief Processes each operation.
   * This loop reads each operation as a string and updates the result 'res'
   * based on whether the operation is an increment or a decrement.
   */
  for (int i = 0; i < n; i++) {
    string op;
    cin >> op;
    /**
     * @brief Updates the result based on the operation.
     * It checks for both prefix and postfix increment and decrement operations
     * and modifies the result accordingly.
     */
    if (op == "++X" || op == "X++")
      res++;
    if (op == "--X" || op == "X--")
      res--;
  }

  // Print the final result.
  cout << res;

  return 0;
}