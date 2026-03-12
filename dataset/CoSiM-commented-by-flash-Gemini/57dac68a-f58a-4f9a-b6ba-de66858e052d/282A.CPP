/**
 * @57dac68a-f58a-4f9a-b6ba-de66858e052d/282A.CPP
 * @brief This program simulates a simple counter by processing a series of commands.
 * Each command, represented as a short C-style string, either increments or decrements a central variable 'x'.
 * The final value of 'x' is then printed.
 *
 * Algorithm: Iterative parsing of fixed-format string commands using `strcmp` for conditional updates.
 * Time Complexity: O(T * L) where T is the number of operations and L is the maximum length of a command string (fixed at 4).
 *                  Effectively O(T) as L is a small constant.
 * Space Complexity: O(1) for storing variables and the command string.
 */

#include <cstring>  // For C-style string manipulation functions like strcmp.
#include <iostream> // For standard input/output operations (cin, cout).

using namespace std; // Uses the standard namespace to avoid prefixing std::.

int main() {
  int t, x = 0; // Declares 't' for the number of operations and 'x' (initialized to 0) as the central variable.
  char str[4];  // Declares a C-style character array to store input command strings (e.g., "X++", requires 4 chars: 3 for command + 1 for null terminator).
  cin >> t;     // Reads the total number of operations 't' from standard input.
  
  /**
   * Block Logic: Processes 't' operation commands.
   * Invariant: 'x' reflects the cumulative effect of all processed commands.
   */
  for (; t > 0; t--) { // Loop 't' times, decrementing 't' in each iteration.
    cin >> str;       // Reads the current operation command string into 'str'.
    
    /**
     * Block Logic: Checks if the command is an increment operation.
     * Pre-condition: 'str' contains a valid command string.
     * Invariant: 'x' is incremented if the command matches "X++" or "++X".
     */
    if ((strcmp(str, "X++") == 0) || (strcmp(str, "++X") == 0))
      x = x + 1; // Increments 'x'.
    
    /**
     * Block Logic: Checks if the command is a decrement operation.
     * Pre-condition: 'str' contains a valid command string.
     * Invariant: 'x' is decremented if the command matches "X--" or "--X".
     */
    if ((strcmp(str, "X--") == 0) || (strcmp(str, "--X") == 0))
      x = x - 1; // Decrements 'x'.
  }
  cout << x << endl; // Prints the final value of 'x' to standard output, followed by a newline.
  return 0;          // Indicates successful program termination.
}
