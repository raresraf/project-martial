/**
 * @498cb9d2-e234-40e4-bc09-7d2a6dce091b/282A.CPP
 * @brief This program simulates a simple variable manipulation process,
 *        incrementing or decrementing an integer variable 'x' based on specific
 *        input statements like "++X", "X++", "--X", or "X--".
 * Algorithm: Iterative processing of input strings, using string comparison to
 *            identify increment/decrement operations.
 * Time Complexity: O(N * L) where N is the number of statements and L is the
 *                  average length of a statement (due to string comparisons).
 * Space Complexity: O(L) for storing the input command string.
 */
#include <bits/stdc++.h>

using namespace std;

int main() {
  // Declare variables:
  // n: The number of statements to process.
  int n;
  // x: The integer variable to be manipulated, initialized to 0.
  int x = 0;
  // command: String to store each input statement.
  string command;
  // Functional Utility: Read the number of statements.
  cin >> n;
  // Block Logic: Iterate 'n' times, processing each statement.
  for (int i = 0; i < n; ++i) {
    // Functional Utility: Read the current statement.
    cin >> command;
    // Pre-condition: Check if the command is an increment operation ("++X" or "X++").
    if (command.compare("++X") == 0 || command.compare("X++") == 0) {
      x++; // Action: Increment 'x'.
    }
    // Pre-condition: Otherwise, check if the command is a decrement operation ("--X" or "X--").
    else if (command.compare("--X") == 0 || command.compare("X--") == 0) {
      x--; // Action: Decrement 'x'.
    }
  }
  // Functional Utility: Output the final value of 'x'.
  cout << x << endl;
  return 0; // Indicate successful program termination.
}