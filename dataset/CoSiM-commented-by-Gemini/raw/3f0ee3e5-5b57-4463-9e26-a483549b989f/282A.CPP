/**
 * @file 282A.CPP
 * @brief Solution for a simple variable manipulation problem.
 * @details This program calculates the final value of a variable 'x' after a series
 *          of increment and decrement operations. It's a common type of problem
 *          in introductory competitive programming.
 *
 * Algorithm:
 * 1. Initialize a variable 'x' to 0.
 * 2. Read the total number of operations, 'n'.
 * 3. Loop 'n' times:
 *    a. Read the operation as a string.
 *    b. If the string contains "++", increment 'x'.
 *    c. If the string contains "--", decrement 'x'.
 * 4. Print the final value of 'x'.
 *
 * Time Complexity: O(n), where 'n' is the number of statements.
 * Space Complexity: O(1), as storage is constant regardless of input size.
 */
#include <iostream>
#include <string>

using namespace std;

/**
 * @brief Main function to execute the program logic.
 * @return 0 on successful execution.
 */
int main() {
  // 'x' is the variable being manipulated, initialized to 0.
  // 'n' is the number of statements to be processed.
  int x = 0, n;
  string statement;
  cin >> n;

  // Loop through each statement, updating 'x' accordingly.
  // Invariant: 'x' holds the cumulative result of operations processed so far.
  while (n--) {
    cin >> statement;
    // Pre-condition: 'statement' is one of "X++", "++X", "X--", or "--X".
    if (statement == "X++" || statement == "++X") {
      x++;
    }
    // Handles both pre- and post-decrement operations.
    if (statement == "X--" || statement == "--X") {
      x--;
    }
  }

  // Output the final value of 'x' after all operations.
  printf("%d\n", x);
  return 0;
}
