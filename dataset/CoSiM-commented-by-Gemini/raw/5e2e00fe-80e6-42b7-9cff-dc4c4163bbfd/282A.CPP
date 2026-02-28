/**
 * @file 282A.CPP
 * @brief Solution for the "Bit++" problem from Codeforces.
 * @details This program calculates the final value of a variable 'X' after a series
 * of increment ("++X", "X++") and decrement ("--X", "X--") operations. This specific
 * solution identifies the operation by inspecting the character at index 1 of the
 * operation string.
 *
 * @algorithm Fixed-Index Character Check
 * The key insight is that for all possible operation strings, the character at index 1
 * is sufficient to determine the operation type:
 * - "++X": `input[1]` is '+'
 * - "X++": `input[1]` is '+'
 * - "--X": `input[1]` is '-'
 * - "X--": `input[1]` is '-'
 * The algorithm leverages this by initializing a counter to 0, looping through the
 * operations, and incrementing or decrementing the counter based on the character at index 1.
 *
 * @complexity
 * - Time: O(N), where N is the number of statements, as each statement is processed once
 *   with a constant-time check.
 * - Space: O(1), as it only stores the current operation string (which is short) and
 *   a few variables.
 */

#include <bits/stdc++.h>
using namespace std;

int main() {
  // `testcases`: the number of operations to perform.
  // `x`: the counter variable, initialized to 0.
  int testcases, x = 0;
  // `input`: stores the operation string (e.g., "++X").
  string input;
  
  cin >> testcases;
  
  /**
   * @brief Process each of the `testcases` operations.
   */
  for (int i = 0; i < testcases; i++) {
    cin >> input;
    
    /**
     * @brief Determine the operation by checking the character at index 1.
     * This is a highly efficient check as the middle character is always
     * '+' for an increment and '-' for a decrement.
     */
    // Pre-condition: Check if the operation is an increment.
    if (input[1] == '+') {
      x++;
    } else {
      x--;
    }
  }
  
  // Print the final value of x.
  cout << x;

  return 0; // Indicate successful execution.
}
