/**
 * @58bfb367-dc13-43de-b01e-3b2a64f51024/282A.CPP
 * @brief This program simulates a simple counter variable 'x' that is incremented or decremented
 * based on a series of input command strings. Each command is a 3-character string
 * representing pre-increment/decrement or post-increment/decrement operations.
 *
 * Algorithm: Iterative parsing of fixed-format string commands with explicit character checks
 * to determine and apply increment/decrement operations.
 * Time Complexity: O(N * L) where N is the number of operations and L is the length of the command string (fixed at 3).
 *                  Effectively O(N) as L is a small constant.
 * Space Complexity: O(1) for storing variables and the command string.
 */

#include <iostream> // For standard input/output operations (cin, cout).
#include <string>   // For using std::string.
#include <cctype>   // For character handling functions like isalpha.

using namespace std; // Uses the standard namespace to avoid prefixing std::.

// Type definitions for convenience, commonly used in competitive programming.
typedef long long int lli;
typedef long long ll;
typedef unsigned long long int ulli;

int main(int argc, char const *argv[]) {
  lli n; // Declares 'n', representing the total number of operations to process.
  cin >> n; // Reads the number of operations from standard input.
  lli x = 0; // Declares and initializes 'x' to 0, the variable that will be incremented or decremented.
  
  /**
   * Block Logic: Processes 'n' operation commands. For each command, it determines
   * whether to increment or decrement 'x' based on specific string patterns.
   * Invariant: 'x' accurately reflects the cumulative effect of all processed commands.
   */
  while (n--) { // Loop 'n' times, decrementing 'n' in each iteration.
    string a;   // Declares a string variable to store the current command.
    cin >> a;   // Reads the command string from standard input.
    
    /**
     * Block Logic: Checks for increment operations ("X++" or "++X").
     * Pre-condition: 'a' is a 3-character string.
     * Invariant: 'x' is incremented if the command matches one of the increment patterns.
     * Inline: `isalpha(a[0])` or `isalpha(a[2])` ensures the character is a letter (i.e., 'X').
     */
    if (isalpha(a[0]) && (a[1]) == '+' && a[2] == '+') { // Checks for "X++" pattern (e.g., "X++").
      x++; // Increments 'x'.
    } else if (isalpha(a[2]) && (a[0] == '+') && (a[1] == '+')) { // Checks for "++X" pattern (e.g., "++X").
      x++; // Increments 'x'.
    } 
    /**
     * Block Logic: Checks for decrement operations ("X--" or "--X").
     * Pre-condition: 'a' is a 3-character string.
     * Invariant: 'x' is decremented if the command matches one of the decrement patterns.
     */
    else if (isalpha(a[2]) && (a[0] == '-') && (a[1] == '-')) // Checks for "--X" pattern (e.g., "--X").
      x--; // Decrements 'x'.
    else if (isalpha(a[0]) && (a[1] == '-') && (a[2] == '-')) // Checks for "X--" pattern (e.g., "X--").
      x--; // Decrements 'x'.
  }
  cout << x << endl; // Prints the final value of 'x' to standard output, followed by a newline.
  return 0;          // Indicates successful program termination.
}