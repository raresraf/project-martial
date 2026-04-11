/**
 * @file 282A.CPP
 * @brief This program simulates a simple counter manipulation based on a series of operations.
 * It reads an integer N, then N operations. Each operation either increments or decrements a counter X.
 * The final value of X is printed.
 *
 * Algorithm: Simple counter with conditional increment/decrement.
 * Time Complexity: O(N) where N is the number of operations.
 * Space Complexity: O(1)
 */

#include <iostream>
#include <string>

using namespace std;

int main() {
  // Functional Utility: Declares an integer 'n' to store the number of operations.
  int n;
  // Functional Utility: Initializes the counter 'X' to 0. This variable will be
  //                      modified by the operations.
  int X = 0;
  // Block Logic: Reads the total number of operations 'n' from standard input.
  cin >> n;
  /**
   * Block Logic: Iterates 'n' times, processing one operation in each iteration.
   * Invariant: At the start of each iteration, 'i' represents the remaining number of operations to process.
   */
  for (int i = n; i > 0; i--) {
    // Functional Utility: Declares a string 'op' to store the current operation.
    string op;
    // Block Logic: Reads the current operation string from standard input.
    cin >> op;
    /**
     * Block Logic: Checks if the operation string contains "++".
     * Pre-condition: 'op' contains the operation string.
     * Invariant: If "++" is found, the counter 'X' is incremented.
     */
    if (op.find("++") != string::npos) {
      X++;
    } 
    /**
     * Block Logic: If "++" is not found, it implies the operation is "--".
     * Pre-condition: 'op' does not contain "++".
     * Invariant: The counter 'X' is decremented.
     */
    else {
      X--;
    }
  }
  // Output the final value of the counter 'X'.
  cout << X << endl;
  return 0;
}
