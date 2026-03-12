/**
 * @5e2e00fe-80e6-42b7-9cff-dc4c4163bbfd/282A.CPP
 * @brief This program simulates a simple counter variable 'x' that is modified
 * by a series of input commands. Each command is a string representing either
 * an increment or a decrement operation (e.g., "++X", "X++", "--X", "X--").
 * The program identifies the operation based on the character at the second position
 * of the command string.
 *
 * Algorithm: Iterative parsing of string commands with character-based conditional updates.
 * Time Complexity: O(N) where N is the number of operations.
 * Space Complexity: O(1) for storing variables and the command string.
 */

#include <bits/stdc++.h> // A common header in competitive programming; includes iostream, string.
using namespace std;     // Uses the standard namespace.

int main() {
  int testcases, x = 0; // Declares 'testcases' for the number of operations and 'x' (the counter) initialized to 0.
  string input;          // Declares a string variable to store each command.
  cin >> testcases;      // Reads the number of operations from standard input.
  
  /**
   * Block Logic: Processes `testcases` operation commands. For each command, it reads a string
   * and modifies 'x' accordingly.
   * Invariant: 'x' accurately reflects the cumulative effect of all processed commands.
   */
  for (int i = 0; i < testcases; i++) { // Loops `testcases` times, once for each operation.
    cin >> input; // Reads the command string from standard input.
    
    /**
     * Block Logic: Determines whether the command is an increment or decrement operation.
     * Pre-condition: 'input' is a valid command string.
     * Invariant: 'x' is either incremented or decremented by 1.
     * Inline: Checks the character at index 1 of the string. This works for patterns like "++X" and "X++".
     */
    if (input[1] == '+') { 
      x++; // Increments 'x'.
    } else { // Inline: Otherwise, it's assumed to be a decrement operation.
      x--; // Decrements 'x'.
    }
  }
  cout << x; // Prints the final value of 'x'.

  return 0; // Indicates successful program termination.
}
