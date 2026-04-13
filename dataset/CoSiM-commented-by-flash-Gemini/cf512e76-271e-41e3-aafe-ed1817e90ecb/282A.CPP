/**
 * @file 282A.CPP
 * @brief This program simulates a basic programming language with a single integer variable 'x'
 * and two operations: increment ('++') and decrement ('--').
 * The operations can be represented in multiple forms (e.g., "++X", "X++", "--X", "X--").
 * Algorithm: Parses operation strings and applies increment/decrement to a counter.
 * Time Complexity: O(N * L) where N is the number of statements and L is the maximum length of an operation string.
 * Space Complexity: O(L) to store the operation string.
 */

#include <bits/stdc++.h>
using namespace std;
int main() {
  long long int n, i, j, x = 0;
  // Reads the number of statements to process.
  cin >> n;
  char a[1000]; // Buffer to store the operation string.
  /**
   * Block Logic: Iterates 'n' times to process each statement.
   * Invariant: 'x' holds the cumulative result of operations processed so far.
   */
  for (i = 0; i < n; i++) {
    // Reads the current operation statement.
    cin >> a;
    /**
     * Conditional Logic: Determines whether to increment or decrement 'x' based on the operation string.
     * It checks if the first or third character is '+' to identify an increment operation.
     * Otherwise, it's assumed to be a decrement operation.
     */
    if (a[0] == '+' || a[2] == '+')
      x += 1;
    else
      x -= 1;
  }
  // Prints the final value of 'x' after all operations are processed.
  cout << x;
  // Standard program exit.
  return 0;
}