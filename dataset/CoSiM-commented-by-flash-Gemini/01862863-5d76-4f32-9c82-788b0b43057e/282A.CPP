/**
 * @file 282A.CPP
 * @brief Solution for a programming problem involving a single variable `X`
 *        and operations to increment or decrement it. The task is to compute
 *        the final value of `X` after a sequence of operations.
 *
 * Algorithm: Simple iterative counter.
 * Time Complexity: $O(N)$ where N is the number of operations.
 * Space Complexity: $O(1)$ as only a few variables are stored.
 */

#include <iostream>
#include <string> // Required for string type
using namespace std;

/**
 * @brief Main function to read operations and compute the final value of a counter.
 * Functional Utility: Parses a sequence of increment/decrement operations
 *                     and applies them to a central counter, demonstrating basic
 *                     input processing and conditional logic.
 * @return 0 on successful execution.
 */
int main() {
  int n; // Declares an integer to store the number of operations.
  string s; // Declares a string to store each operation.
  cin >> n; // Reads the total number of operations from standard input.
  int c = 0; // Initializes the counter 'c' to zero.
  /**
   * @brief Block Logic: Processes each of the 'n' operations.
   * Invariant: In each iteration, one operation is read and applied to `c`.
   * Precondition: `n` holds the remaining number of operations to process.
   * Postcondition: `n` is decremented, and `c` is updated based on the operation.
   */
  while (n--) {
    cin >> s; // Reads the current operation string.
    /**
     * @brief Block Logic: Determines whether to increment or decrement the counter.
     * Precondition: `s` contains a valid operation string ("++X", "X++", "--X", or "X--").
     * Postcondition: `c` is either incremented or decremented by 1.
     */
    if (s == "++X" || s == "X++")
      c += 1; // Increments 'c' if the operation is an increment.
    else
      c -= 1; // Decrements 'c' if the operation is a decrement.
  }
  cout << c << endl; // Prints the final value of the counter 'c'.
  return 0; // Indicates successful program termination.
}