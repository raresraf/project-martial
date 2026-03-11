/**
 * @file 282A.CPP
 * @brief Solution to Codeforces Problem 282A: "Bit++".
 *        This program simulates a simplified programming language with a single integer variable `X`,
 *        initialized to 0. It processes a series of statements, each either incrementing (`++X`, `X++`)
 *        or decrementing (`--X`, `X--`) the variable. The final value of `X` is then printed.
 *
 * Algorithm: Iterative statement processing. The program reads the number of statements, then iterates
 *            through each statement. It checks if the statement contains "++" to determine an increment
 *            operation, otherwise it's assumed to be a decrement operation.
 * Time Complexity: O(N * L) where N is the number of statements and L is the average length of a statement string.
 *                  Due to string comparison operations.
 * Space Complexity: O(L_max) to store the longest statement string.
 */
#include <bits/stdc++.h>
using namespace std;

int main() {
  /**
   * Functional Utility: Declaration and initialization of variables.
   * `n`: Stores the total number of statements to process.
   * `x`: The integer variable that is incremented or decremented; initialized to 0.
   * `word`: Stores the current statement string read from input.
   */
  int n, x = 0;
  string word;
  cin >> n; // Input: Read the number of statements.

  /**
   * Block Logic: Iterates `n` times, processing one statement in each iteration.
   * Pre-condition: `n` is the total number of statements.
   * Invariant: `word` contains the current statement to be evaluated and `x` reflects
   *            the cumulative effect of previous operations.
   */
  for (int i = 0; i < n; i++) {
    cin >> word; // Input: Read the current statement.
    /**
     * Conditional Logic: Checks if the statement is an increment operation (`++X` or `X++`).
     * If true, increment `x`. Otherwise, it's a decrement operation (`--X` or `X--`), so decrement `x`.
     */
    if (word == "++X" || word == "X++")
      x++; // Increment x.
    else
      x--; // Decrement x.
  }

  // Output: Print the final value of x, followed by a newline.
  cout << x << endl;

  return 0;
}