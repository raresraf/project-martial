/**
 * @5dabbce1-3dd7-4b77-a341-37398c0b8af6/282A.CPP
 * @brief This program simulates a simple counter variable 'c' that is modified
 * by a series of input command strings. Each command, typically a 3-character string
 * (e.g., "++X", "X++", "--X", "X--"), either increments or decrements the variable.
 * The program identifies increment operations by the presence of a '+' character.
 *
 * Algorithm: Iterative parsing of string commands using `std::string::find` for conditional updates.
 * Time Complexity: O(N * L) where N is the number of operations and L is the fixed length of the command string.
 *                  Effectively O(N) as L is a small constant.
 * Space Complexity: O(1) for storing variables and the command string.
 */

#include <iostream> // For standard input/output operations (cin, cout).
#include <string>   // For using std::string.

using namespace std; // Uses the standard namespace.

int main() {
  int n, c = 0; // Declares 'n' for the number of operations and 'c' (the counter) initialized to 0.
  cin >> n;     // Reads the number of operations from standard input.

  /**
   * Block Logic: Processes 'n' operation commands. For each command, it reads a string
   * and modifies 'c' accordingly.
   * Invariant: 'c' accurately reflects the cumulative effect of all processed commands.
   */
  while (n--) { // Loops 'n' times, decrementing 'n' in each iteration.
    string s;   // Declares a string variable to store the current command.
    cin >> s;   // Reads the command string from standard input.

    /**
     * Block Logic: Determines whether the command is an increment or decrement operation.
     * Pre-condition: 's' contains a valid command string.
     * Invariant: 'c' is either incremented or decremented by 1.
     * Inline: `s.find('+') != string::npos` checks if the character '+' exists anywhere in the string.
     *         If it does, it's an increment operation; otherwise, it's a decrement.
     */
    if (s.find('+') != string::npos) 
      c++; // Increments 'c'.
    else
      c--; // Decrements 'c'.
  }

  cout << c << endl; // Prints the final value of 'c' to standard output, followed by a newline.

  return 0; // Indicates successful program termination.
}
