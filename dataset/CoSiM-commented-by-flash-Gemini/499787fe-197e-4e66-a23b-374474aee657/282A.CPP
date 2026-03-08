/**
 * @499787fe-197e-4e66-a23b-374474aee657/282A.CPP
 * @brief This program simulates a simple variable manipulation process,
 *        incrementing or decrementing an integer variable 'x' based on the
 *        presence of '+' or '-' characters within input statements.
 * Algorithm: Iterative processing of input strings, searching for specific
 *            characters to determine increment/decrement operations.
 * Time Complexity: O(N * L) where N is the number of statements and L is the
 *                  average length of a statement (due to string iteration).
 * Space Complexity: O(L) for storing the input string.
 */
#include <bits/stdc++.h>
using namespace std;
int main() {
  // Declare variables:
  // n: The number of statements to process.
  // x: The integer variable to be manipulated, initialized to 0.
  int n, x = 0;
  // Functional Utility: Read the number of statements.
  cin >> n;
  // Block Logic: Iterate 'n' times, processing each statement.
  while (n--) {
    string s; // Declare a string 's' to store each input statement.
    // Functional Utility: Read the current statement.
    cin >> s;
    int l = s.length(); // Get the length of the statement string.
    // Block Logic: Iterate through the characters of the statement to find '+' or '-'.
    for (int i = 0; i < l; i++) {
      if (s[i] == '+') { // Pre-condition: If a '+' character is found.
        x++;             // Action: Increment 'x'.
        break;           // Break from inner loop as operation is determined.
      } else if (s[i] == '-') { // Pre-condition: If a '-' character is found.
        x--;                     // Action: Decrement 'x'.
        break;                   // Break from inner loop as operation is determined.
      }
    }
  }
  // Functional Utility: Output the final value of 'x'.
  cout << x;
  return 0; // Indicate successful program termination.
}
