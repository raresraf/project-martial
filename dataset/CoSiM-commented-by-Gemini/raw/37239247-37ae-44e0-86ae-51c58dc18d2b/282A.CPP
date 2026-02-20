/**
 * @file 282A.CPP
 * @brief A concise solution to the "Bit++" problem.
 * @note This solution uses a clever trick: it checks only the second character
 * of the input string to determine whether the operation is an increment or
 * a decrement. For all valid inputs ("++X", "X++", "--X", "X--"), the character
 * at index 1 is unique to the operation type ('+' or '-').
 *
 * This program simulates a simple programming language with one variable, 'X',
 * initialized to 0, and two operations: increment and decrement. It calculates
 * the final value of X after a series of operations.
 *
 * Algorithm:
 * 1. Read the number of statements, 't'.
 * 2. Initialize a counter 'k' to 0.
 * 3. Loop 't' times:
 *    a. Read the operation string.
 *    b. If the second character is '-', decrement k.
 *    c. Otherwise, increment k.
 * 4. Print the final value of k.
 *
 * Time Complexity: O(T), where T is the number of statements.
 * Space Complexity: O(1), as the string buffer is reused.
 */
#include <bits/stdc++.h>

using namespace std;

string s; // To store the operation string.
int t;    // Number of statements.
// k: The variable being operated on, equivalent to 'X'. Initialized to 0 by default as a global.
int k;

main() {

  cin >> t;

  /**
   * Block Logic: Main processing loop.
   * Iterates 't' times, processing one statement per iteration.
   */
  while (t--) {

    cin >> s;

    /**
     * Block Logic: Operation check based on the second character.
     * For "++X" and "X++", s[1] is '+'.
     * For "--X" and "X--", s[1] is '-'.
     * This provides a simple and efficient way to determine the operation.
     */
    if (s[1] == '-')
      k--;
    else
      k++;
  }

  // Print the final value of the variable 'k'.
  cout << k;
}
