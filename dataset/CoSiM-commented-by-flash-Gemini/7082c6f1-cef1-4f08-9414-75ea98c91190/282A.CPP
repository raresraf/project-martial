/**
 * @file 282A.CPP
 * @brief Solution to Codeforces Problem 282A: "Bit++".
 * Simulates a programming language with a single variable `X` and `n` operations.
 * Operations can be `++X`, `X++`, `--X`, or `X--`. The task is to find the final value of `X`.
 * Algorithm: Initialize `X` to 0. For each operation, parse the string and increment or
 * decrement `X` accordingly.
 * Time Complexity: O(N), where N is the number of operations, as each operation is processed
 *                  in constant time (string comparison and arithmetic).
 * Space Complexity: O(1) for storing variables, as the operation string length is constant.
 */
#include "string"   // Required for string manipulation.
#include <iostream> // Required for input/output operations (cin, cout).
using namespace std;

/**
 * @brief Main function to solve the "Bit++" problem.
 * Reads the number of operations, then processes each operation to update a variable 'X',
 * and finally prints the variable's value.
 *
 * @param argc The number of command-line arguments.
 * @param argv An array of command-line argument strings.
 */
int main(int argc, const char *argv[]) {
  int n;      // Stores the total number of operations.
  int res = 0; // Stores the current value of the variable X, initialized to 0.
  cin >> n;   // Reads the number of operations.

  /**
   * Block Logic: Processes each operation one by one.
   * Pre-condition: 'n' specifies the number of operations. 'res' is 0.
   * Invariant: After each iteration, 'res' reflects the value of X after 'i+1' operations.
   */
  for (int i = 0; i < n; i++) {
    string op; // Stores the string representation of the current operation (e.g., "++X").
    cin >> op; // Reads the operation string.

    /**
     * Block Logic: Checks if the operation is an increment operation.
     * If 'op' is "++X" or "X++", the value of 'res' is incremented.
     */
    if (op == "++X" || op == "X++") {
      res++;
    }
    /**
     * Block Logic: Checks if the operation is a decrement operation.
     * If 'op' is "--X" or "X--", the value of 'res' is decremented.
     */
    if (op == "--X" || op == "X--") { // Note: 'else if' could be used for minor optimization, but 'if' works.
      res--;
    }
  }

  cout << res; // Prints the final value of 'res' (which is X).

  return 0;
}
