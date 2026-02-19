/**
 * @file 282A.CPP
 * @brief Solution to the "Bit++" problem from Codeforces.
 * @details This program simulates a simple programming language with one variable 'x' and two operations:
 *          '++' which increments the variable, and '--' which decrements it. The program calculates
 *          the final value of 'x' after a series of operations.
 *
 * Algorithm:
 * 1. Initialize a counter `ans` to 0, representing the initial value of 'x'.
 * 2. Read the total number of operations, `n`.
 * 3. Loop `n` times to process each operation:
 *    a. Read the operation as a string `s`.
 *    b. Check for the presence of a '+' character at the first or third position (e.g., "++X" or "X++").
 *       If a '+' is found, increment `ans`.
 *    c. Otherwise, the operation must contain a '-', so decrement `ans`.
 * 4. After all operations are processed, print the final value of `ans`.
 *
 * Time Complexity: O(N), where N is the number of operations, as each operation is processed in constant time.
 * Space Complexity: O(1), as the space used does not scale with the number of operations (the input string is reused).
 */
#include <bits/stdc++.h>
using namespace std;
int main() {
  /**
   * @brief n: the number of statements to execute.
   *        ans: the final value of the variable 'x', initialized to 0.
   */
  int n, ans = 0;
  cin >> n;

  /**
   * @brief Main loop to process each of the n statements.
   * Invariant: `ans` holds the value of 'x' after processing the statements so far.
   */
  while (n--) {
    string s;
    cin >> s;
    /**
     * @brief Check the operation type. The statements are of the form "++X", "X++", "--X", or "X--".
     *        A '+' at index 0 or 2 indicates an increment operation.
     *        Otherwise, it's a decrement operation.
     */
    if (s[0] == '+' || s[2] == '+')
      ans++;
    else
      ans--;
  }
  
  // Output the final value of 'x'.
  cout << ans << "\n";
}
