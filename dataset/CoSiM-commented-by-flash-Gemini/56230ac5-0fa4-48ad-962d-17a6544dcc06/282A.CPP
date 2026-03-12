/**
 * @56230ac5-0fa4-48ad-962d-17a6544dcc06/282A.CPP
 * @brief This program simulates a simple counter operation based on a series of commands.
 * Each command, represented as a string, either increments or decrements a central variable.
 *
 * Algorithm: Iterative processing of string commands to modify a single integer counter.
 * Time Complexity: O(N) where N is the number of operations.
 * Space Complexity: O(1) for storing the counter and input string.
 */

#include <iostream> // Includes the standard input/output stream library.
#include <string> // Includes the string library for using std::string.

using namespace std; // Uses the standard namespace to avoid prefixing std::.

int main() {
  int n; // Declares 'n', representing the total number of operations to process.
  int x(0); // Declares and initializes 'x', the central variable that will be incremented or decremented.
  cin >> n; // Reads the total number of operations from standard input.

  string s; // Declares a string variable to store each operation command.
  /**
   * Block Logic: Processes 'n' operation commands. The loop continues until all 'n' operations are processed.
   * Invariant: 'x' accurately reflects the cumulative effect of all processed operations.
   */
  while (n--) { // Loop 'n' times, decrementing 'n' in each iteration.
    cin >> s; // Reads the current operation command string from standard input.
    /**
     * Block Logic: Determines whether to increment or decrement 'x' based on the command string.
     * Pre-condition: 's' contains a valid operation string (e.g., "++X", "X++", "--X", "X--").
     * Invariant: 'x' is either incremented or decremented by 1.
     */
    if (s[1] == '+') { // Checks the character at index 1 of the string.
                       // Inline: If it's a '+', it signifies an increment operation.
      x++; // Increments the variable 'x'.
    } else { // Otherwise (if s[1] is '-'), it signifies a decrement operation.
      x--; // Decrements the variable 'x'.
    }
  }

  cout << x; // Prints the final value of 'x' to standard output.
  return 0; // Indicates successful program termination.
}
