/**
 * @file 282A.CPP
 * @brief Solution for the "Bit++" problem.
 * @details This program simulates a set of operations on a variable 'X',
 * initialized to 0. It processes a series of statements like "++X", "X++",
 * "--X", and "X--" and calculates the final value of X.
 *
 * Algorithm:
 * 1. Initialize a variable, `suma`, to 0. This variable represents X.
 * 2. Read the number of statements, 'n'.
 * 3. Loop 'n' times:
 *    a. Read the statement string.
 *    b. Check the string content:
 *       - If it contains "++", increment `suma`.
 *       - If it contains "--", decrement `suma`.
 * 4. Print the final value of `suma`.
 *
 * Time Complexity: O(N), where N is the number of statements.
 * Space Complexity: O(1) (ignoring the storage for the input string which is reused).
 */
#include <iostream>
using namespace std;

int main() {
  int n;
  cin >> n;
  int suma = 0;
  string s;

  /**
   * Block Logic: Process each statement in a loop.
   * Invariant: `suma` holds the value of X after i-1 iterations.
   */
  for (int i = 1; i <= n; i++) {
    cin >> s;

    // A series of if-statements checks for the operation type.
    // Note: A more optimal approach could check character s[1] ('+' or '-').
    if (s == "++X") {
      suma++;
    }
    if (s == "X++") {
      suma++;
    }
    if (s == "--X") {
      suma--;
    }
    if (s == "X--") {
      suma--;
    }
  }

  cout << suma;

  return 0;
}
