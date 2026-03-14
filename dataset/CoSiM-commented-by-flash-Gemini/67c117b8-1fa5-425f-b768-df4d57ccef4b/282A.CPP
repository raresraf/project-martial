/**
 * @67c117b8-1fa5-425f-b768-df4d57ccef4b/282A.CPP
 * @brief This program solves Codeforces problem 282A, "Bit++".
 * It simulates a simple counter variable (implicitly `res`) that is modified by
 * a series of input command strings. Each command represents either an increment
 * (e.g., "X++", "++X") or a decrement (e.g., "X--", "--X") operation.
 * The program reads the number of commands and processes each one to determine
 * the final value of the counter.
 *
 * Algorithm: Iterative processing of string commands to update an integer counter.
 * String comparison is used to identify increment/decrement operations.
 * Time Complexity: O(N * L) where N is the number of commands and L is the
 * length of the command string (constant, typically 3). Effectively O(N).
 * Space Complexity: O(L) for storing the command string.
 */
// 282A
#include <cstring>  // Included for C-style string functions, though not explicitly used for `std::string` ops.
#include <iostream> // For standard input/output operations (cin, cout).
using namespace std; // Uses the standard namespace.

/**
 * @brief Main function to process a series of increment/decrement commands.
 * Reads the number of operations, then each operation string, updates a counter,
 * and prints the final counter value.
 * @return 0 on successful execution.
 */
int main() {
  int n; // Declares 'n', representing the total number of operations.
  string s; // Declares a string variable to store each operation command.
  int res = 0; // Initializes 'res' (the counter variable) to 0.
  
  cin >> n; // Reads the number of operations from standard input.
  
  /**
   * Block Logic: Processes 'n' operation commands.
   * For each command, it checks if it's an increment or decrement operation
   * and updates the `res` counter accordingly.
   * Invariant: `res` holds the cumulative result of all operations.
   */
  while (n--) { // Loops 'n' times, decrementing 'n' in each iteration.
    cin >> s; // Reads an operation command string (e.g., "X++").
    
    /**
     * Block Logic: Determines if the command is an increment or decrement.
     * Compares the input string `s` against known increment/decrement patterns.
     * Pre-condition: `s` contains a valid operation string.
     * Invariant: `res` is incremented or decremented based on the command.
     */
    if (s == "X++" || s == "++X") // Checks for increment operations.
      res++; // Increments 'res'.
    else if (s == "X--" || s == "--X") // Checks for decrement operations.
      res--; // Decrements 'res'.
  }
  cout << res; // Prints the final value of the counter.
  return 0; // Indicates successful program termination.
}
