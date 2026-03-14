/**
 * @69664ca0-8ea3-4c59-851b-db152a3d4dcf/282A.CPP
 * @brief This program solves Codeforces problem 282A, "Bit++".
 * It simulates a simple counter variable `X` that is modified by a series of
 * input command strings. Each command represents either an increment
 * (e.g., "X++", "++X") or a decrement (e.g., "X--", "--X") operation.
 * The program reads the number of commands and then processes each command
 * to determine the final value of the counter. This solution uses explicit
 * string comparisons against all four possible command formats.
 *
 * Algorithm: Iterative processing of string commands to update an integer counter.
 * String comparison is used to identify increment/decrement operations.
 * Time Complexity: O(N * L) where N is the number of commands and L is the
 * length of the command string (constant, typically 3). Effectively O(N).
 * Space Complexity: O(L) for storing the command string.
 */
#include <iostream> // For standard input/output operations (cin, cout).

using namespace std; // Uses the standard namespace.

/**
 * @brief Main function to process a series of increment/decrement commands.
 * Reads the number of operations, then each operation string, updates a counter,
 * and prints the final counter value.
 * @return 0 on successful execution.
 */
int main() {
  // Declares string variables for all four possible command formats for comparison.
  string str, s1 = "++X", s2 = "X++", s3 = "--X", s4 = "X--"; 
  int n; // Declares 'n', representing the total number of operations.
  int X = 0; // Initializes 'X' (the counter variable) to 0.

  cin >> n; // Reads the number of operations from standard input.

  /**
   * Block Logic: Processes 'n' operation commands.
   * For each command, it compares the input string `str` against known increment
   * patterns (`s1`, `s2`) or decrement patterns (`s3`, `s4`) and updates `X` accordingly.
   * Invariant: `X` holds the cumulative result of all operations.
   */
  for (int i = 0; i < n; i++) { // Loops 'n' times, once for each operation.
    cin >> str; // Reads an operation command string (e.g., "X++").

    // Block Logic: Determines if the command is an increment or decrement.
    // Compares the input string `str` against known increment patterns.
    if (str == s1 || str == s2) { // Checks for increment operations ("++X" or "X++").
      X++; // Increments 'X'.
    } else { // If not an increment operation, it must be a decrement operation.
             // This assumes input will only be one of the four valid patterns.
      X--; // Decrements 'X'.
    }
  }

  cout << X << endl; // Prints the final value of the counter, followed by a newline.

  return 0; // Indicates successful program termination.
}