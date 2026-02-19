/**
 * @file 282A.CPP
 * @brief Solution to the "Bit++" problem from Codeforces.
 * @details This program simulates a simple programming language with one variable 'x' and two operations:
 *          '++' which increments the variable, and '--' which decrements it. The program calculates
 *          the final value of 'x' after a series of operations by performing full string comparisons.
 *
 * Algorithm:
 * 1. Initialize a counter `cnt` to 0, representing the initial value of 'x'.
 * 2. Read the total number of operations, `n`.
 * 3. Loop `n` times to process each operation:
 *    a. Read the operation as a string `x`.
 *    b. Compare the string `x` with "++X" or "X++". If it matches, increment `cnt`.
 *    c. If it's not an increment, compare the string `x` with "--X" or "X--". If it matches, decrement `cnt`.
 * 4. After all operations are processed, print the final value of `cnt`.
 *
 * Time Complexity: O(N), where N is the number of operations. Although string comparison takes time proportional
 * to the string length, the length is constant (3), so each operation is processed in constant time.
 * Space Complexity: O(1), as the space used does not scale with the number of operations (the input string is reused).
 */
#include <bits/stdc++.h>
using namespace std;

int main() {
  /**
   * @brief cnt: the final value of the variable 'x', initialized to 0.
   *        n: the number of statements to execute.
   *        x: a string to store the operation.
   */
  int cnt = 0, n;
  string x;
  cin >> n;

  /**
   * @brief Main loop to process each of the n statements.
   * Invariant: `cnt` holds the value of 'x' after processing the statements so far.
   */
  for (int i = 0; i < n; i++) {
    cin >> x;
    /**
     * @brief Check the operation type using string comparison.
     */
    if (x == "++X" || x == "X++")
      cnt++;
    else if (x == "--X" || x == "X--")
      cnt--;
  }
  // Output the final value of 'x'.
  cout << cnt;
}
