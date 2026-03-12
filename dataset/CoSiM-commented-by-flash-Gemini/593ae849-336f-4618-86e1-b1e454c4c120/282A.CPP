/**
 * @593ae849-336f-4618-86e1-b1e454c4c120/282A.CPP
 * @brief This program simulates a single integer variable 'x' that is modified
 * by a series of input commands. Each command is a string indicating either
 * an increment ("++X", "X++") or a decrement ("--X", "X--") operation.
 * The program outputs the final value of 'x' after all operations.
 *
 * Algorithm: Iterative processing of string commands using direct string comparison for conditional updates.
 * Time Complexity: O(N * L) where N is the number of operations and L is the fixed length of the command string (3).
 *                  Effectively O(N) as L is a small constant.
 * Space Complexity: O(1) for storing variables and the command string.
 */

#include <bits/stdc++.h> // A common header in competitive programming; includes iostream and string.
using namespace std;     // Uses the standard namespace.

int main() {
  int n; // Declares 'n', representing the total number of operations to process.
  cin >> n; // Reads the number of operations from standard input.

  int x = 0; // Declares and initializes 'x' to 0, the variable that will be modified.
  /**
   * Block Logic: Processes 'n' operation commands. For each command, it reads a string
   * and modifies 'x' accordingly.
   * Invariant: 'x' accurately reflects the cumulative effect of all processed commands.
   */
  for (int i = 1; i <= n; i++) { // Loops 'n' times, once for each operation.
    string statement;            // Declares a string variable to store the current command.
    cin >> statement;            // Reads the command string from standard input.

    /**
     * Block Logic: Determines whether the command is an increment or decrement operation.
     * Pre-condition: 'statement' is one of the four valid command strings ("++X", "X++", "--X", "X--").
     * Invariant: 'x' is either incremented or decremented by 1.
     */
    if (statement == "++X" || statement == "X++") { // Inline: Checks if the statement matches an increment pattern.
      x += 1; // Increments 'x'.
    } else if (statement == "--X" || statement == "X--") { // Inline: Checks if the statement matches a decrement pattern.
      x -= 1; // Decrements 'x'.
    }
  }

  cout << x << endl; // Prints the final value of 'x' to standard output, followed by a newline.

  return 0; // Indicates successful program termination.
}
