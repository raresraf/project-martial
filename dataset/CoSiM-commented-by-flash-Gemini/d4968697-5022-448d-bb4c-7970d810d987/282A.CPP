/**
 * @file 282A.CPP
 * @brief This program simulates a simple variable manipulation process.
 * The program reads a series of operations and applies them to a single integer variable, 'x'.
 * Each operation either increments or decrements 'x'.
 * Algorithm: Iterative processing of string operations.
 * Time Complexity: O(N) where N is the number of operations.
 * Space Complexity: O(1) as only a few variables are used.
 */

#include <iostream>

using namespace std;

/**
 * @brief Main function to execute the variable manipulation simulation.
 * It reads the number of operations, then processes each operation
 * to update a central variable 'x', finally printing its value.
 * @return 0 on successful execution.
 */
int main() {
  // @var n: Integer representing the total number of operations to be performed.
  // @var x: Integer variable that undergoes increment/decrement operations. Initialized to 0.
  // @var op: String to store the current operation (e.g., "X++", "++X", "X--", "--X").
  int n = 0, x = 0;
  string op = "";
  // Read the total number of operations from standard input.
  cin >> n;
  /**
   * @brief Loop through each operation.
   * Invariant: At the start of each iteration, 'x' holds the result of all previously processed operations.
   * Pre-condition: 'n' operations are pending to be read from input.
   */
  for (int i = 0; i < n; i++) {
    // Read the current operation string.
    cin >> op;
    // Check if the operation is an increment ("X++" or "++X").
    if (op == "X++" || op == "++X") {
      // Increment the variable 'x'.
      x++;
    }
    // Else, check if the operation is a decrement ("X--" or "--X").
    else if (op == "X--" || op == "--X") {
      // Decrement the variable 'x'.
      x--;
    }
  }
  // Output the final value of 'x' to standard output.
  cout << x << endl;
  return 0;
}
