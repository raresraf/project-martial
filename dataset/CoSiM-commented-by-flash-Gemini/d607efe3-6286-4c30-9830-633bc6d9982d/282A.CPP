/**
 * @file 282A.CPP
 * @brief This program simulates a simple variable manipulation process.
 * It reads a series of operations (e.g., "++X", "X--") and applies them
 * to an integer variable, 'x', finally printing its value.
 * Algorithm: Iterative processing of string operations.
 * Time Complexity: O(N) where N is the number of operations.
 * Space Complexity: O(1) as only a few variables are used.
 */

#include <iostream> // Required for input/output operations (cin, cout).
using namespace std;

/**
 * @brief Main function to execute the variable manipulation simulation.
 * It reads the number of operations, then processes each operation
 * to update a central variable 'x', finally printing its value.
 *
 * @param argc The number of command-line arguments (unused).
 * @param argv An array of command-line argument strings (unused).
 * @return 0 on successful execution.
 */
int main(int argc, char const *argv[]) {
  // @var n: Integer representing the total number of operations to be performed.
  int n;
  // Read the total number of operations from standard input.
  cin >> n;
  // @var x: Integer variable that undergoes increment/decrement operations. Initialized to 0.
  int x = 0;
  /**
   * @brief Loop through each operation.
   * The loop continues `n` times, decrementing `n` in each iteration.
   * Invariant: At the start of each iteration, 'x' holds the result of all previously processed operations.
   * Pre-condition: 'n' operations are pending to be read from input.
   */
  while (n--) {
    // @var str: String to store the current operation (e.g., "X++", "++X", "X--", "--X").
    string str;
    // Read the current operation string from standard input.
    cin >> str;
    // Check the second character of the operation string to determine if it's an increment or decrement.
    // This handles both "++X", "X++", "--X", and "X--" forms.
    if (str[1] == '+')
      x++; // If the second character is '+', increment 'x'.
    else if (str[1] == '-')
      x--; // If the second character is '-', decrement 'x'.
  }
  // Output the final value of 'x' to standard output.
  cout << x;
  return 0;
}