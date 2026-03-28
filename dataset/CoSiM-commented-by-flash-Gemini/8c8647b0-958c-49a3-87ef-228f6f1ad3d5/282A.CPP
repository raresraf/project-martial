/**
 * @file 282A.CPP
 * @brief Solution for a problem involving a sequence of increment and decrement operations.
 *
 * This program reads an integer N, followed by N operations. Each operation is a string
 * that modifies a central counter 'x'. The goal is to compute the final value of 'x'.
 * Examples of operations include "++X" and "--X".
 *
 * Algorithm: Iterative processing of operations, interpreting the second character
 *            of each operation string to determine increment or decrement.
 * Time Complexity: O(N * L), where N is the number of operations and L is the length
 *                  of the operation string. Since L is constant (typically 3),
 *                  this simplifies to O(N).
 * Space Complexity: O(L) for storing the input string.
 */
#include <bits/stdc++.h>
using namespace std;
int main() {
  // Declare 'n' for the number of operations and 'x' as the counter, initialized to 0.
  int n, x = 0;
  string str; // Declare 'str' to store each operation string.
  // Read the total number of operations 'n'.
  cin >> n;
  /**
   * Block Logic: Iterates 'n' times, processing each operation string.
   * Invariant: In each iteration, one operation string is read and its effect
   *            (increment or decrement) is applied to 'x'.
   */
  for (int i = 0; i < n; i++) {
    // Read the current operation string.
    cin >> str;
    /**
     * Block Logic: Checks the second character of the operation string to determine
     *            whether to increment 'x'.
     * Pre-condition: 'str' contains a valid operation string (e.g., "++X", "--X").
     */
    if (str[1] == '+') {
      // Increment 'x' if the second character is '+'.
      x++;
    }
    /**
     * Block Logic: Checks the second character of the operation string to determine
     *            whether to decrement 'x'.
     * Pre-condition: 'str' contains a valid operation string (e.g., "++X", "--X").
     */
    if (str[1] == '-') {
      // Decrement 'x' if the second character is '-'.
      x--;
    }
  }
  // Output the final computed value of 'x'.
  cout << x;
  return 0;
}
