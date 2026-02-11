/**
 * @file 282A.CPP
 * @brief Simulates a series of increment and decrement operations.
 * @details This program calculates the final value of a variable 'x' after a series of operations.
 * The operations are given as strings, such as "++X", "X++", "--X", or "X--". The initial value of x is 0.
 * This is a solution to the "Bit++" problem on Codeforces.
 *
 * @see https://codeforces.com/problemset/problem/282/A
 *
 * Algorithm:
 * 1. Initialize a counter 'ans' to 0.
 * 2. Read the total number of operations, 'n'.
 * 3. Loop 'n' times:
 *    a. Read the operation string.
 *    b. If the string contains a '+' character, increment 'ans'.
 *    c. Otherwise (it contains a '-'), decrement 'ans'.
 * 4. Print the final value of 'ans'.
 *
 * Time Complexity: O(N), where N is the number of operations, as we process each operation once.
 * Space Complexity: O(1), as the space used does not scale with the number of operations.
 */

#include <bits/stdc++.h>
using namespace std;

/**
 * @brief Macro to optimize I/O operations in C++.
 * @details This is a common competitive programming technique to speed up cin/cout
 * by decoupling C++ streams from C stdio and disabling synchronization.
 */
#define FAST_INP                                                               \
  ios_base::sync_with_stdio(false);                                            \
  cin.tie(NULL)

int main() {
  FAST_INP;
  // n: the number of operations.
  // ans: the final value after all operations, initialized to 0.
  int n, ans = 0;
  cin >> n;

  /**
   * @brief Main simulation loop that processes each operation.
   *
   * Pre-condition: 'n' is the number of operations to be processed.
   * Invariant: At the end of each iteration, 'ans' reflects the value after one more operation.
   */
  while (n--) {
    string s;
    cin >> s;
    /**
     * @brief Determines whether to increment or decrement.
     *
     * Logic: Any increment operation ("++X" or "X++") must contain a '+'.
     * Any decrement operation ("--X" or "X--") must contain a '-'.
     * The presence of a '+' is sufficient to identify an increment.
     */
    if (count(s.begin(), s.end(), '+'))
      ans++;
    else
      ans--;
  }

  // Output the final result.
  cout << ans << "\n";
  return 0;
}
