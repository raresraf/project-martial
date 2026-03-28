/**
 * @file 282A.CPP
 * @brief Solution for a problem involving a sequence of increment and decrement operations
 *        on a variable, where operations are provided as strings.
 *
 * This program reads an integer 'ops' representing the number of operations to perform.
 * For each operation, it reads a string ('compute') which can be "X++", "++X", "X--", or "--X".
 * It maintains a counter 'x' and modifies it based on the operation string.
 * Finally, it prints the resultant value of 'x' after all operations.
 *
 * Algorithm: Iterative processing of string-based operations, applying conditional
 *            increment or decrement based on string content.
 * Time Complexity: O(N * L), where N is the number of operations and L is the length
 *                  of an operation string (constant, e.g., 3). This simplifies to O(N).
 * Space Complexity: O(L) for storing the operation string.
 */
#include <cstring> // Included but not explicitly used in this snippet.
#include <iostream> // Required for std::cin, std::cout.
#include <string>   // Required for std::string.
using namespace std; // Using the standard namespace.

// Note: The 'main' function signature without 'int' return type is a compiler extension.
main() {
  // Declare 'ops' for the number of operations, and 'x' as the counter, initialized to 0.
  int ops, x = 0;
  string compute; // Declare 'compute' to store each operation string.
  // Read the total number of operations 'ops'.
  cin >> ops;
  /**
   * Block Logic: Loop through each of the 'ops' operations to modify 'x'.
   * Invariant: 'ops' decreases with each iteration; 'x' is updated based on the current operation.
   */
  while (ops) {
    // Read the current operation string.
    cin >> compute;
    /**
     * Block Logic: Determines whether to increment 'x' based on the operation string.
     * Pre-condition: 'compute' contains a valid increment/decrement operation string.
     */
    if (compute == "X++" || compute == "++X") {
      // Increment 'x' if the operation indicates an addition (post-increment or pre-increment).
      x++;
    } else {
      // Otherwise (if the operation is "X--" or "--X"), decrement 'x'.
      x--;
    }
    // Decrement the operations counter.
    ops--;
  }
  // Output the final computed value of 'x'.
  cout << x;
  // The 'main' function implicitly returns 0 in C++ if no return statement is present.
}
