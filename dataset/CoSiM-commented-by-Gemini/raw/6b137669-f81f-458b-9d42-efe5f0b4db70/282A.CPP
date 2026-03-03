
#include <iostream>
using namespace std;

/**
 * @file 282A.CPP
 * @brief Solution to Codeforces problem 282A, "Bit++".
 * @details The program simulates a series of increment and decrement operations on a
 * single variable, `X`, which starts at 0. The operations are given as strings
 * like "++X", "X++", "--X", or "X--".
 *
 * Algorithm:
 * The program initializes a counter `x` to 0. It then reads the number of
 * operations to be performed. In a loop, it reads each operation as a string.
 * A key observation is that all increment operations ("++X", "X++") contain a
 * '+' character, and all decrement operations ("--X", "X--") contain a '-'
 * character. Even more specifically, the character at the middle index (index 1)
 * is always '+' for an increment and '-' for a decrement. The program uses this
 * insight to check just the middle character to determine whether to increment
 * or decrement the counter `x`. Finally, it prints the final value of `x`.
 *
 * Time Complexity: O(N), where N is the number of operations, as it processes
 * each operation string once.
 * Space Complexity: O(1), ignoring the storage for the input string which is
 * reused in each loop iteration.
 */
int main() {
  // n: the number of statements to execute.
  // x: the variable, initialized to 0.
  int n, x = 0;
  cin >> n;

  // opr: string to hold the current operation (e.g., "++X").
  string opr;

  /**
   * @brief Process each operation statement.
   *
   * Invariant: `x` holds the accumulated value from the operations processed so far.
   */
  for (int i = 0; i < n; i++) {
    cin >> opr;
    /**
     * @brief Check the middle character to determine the operation type.
     *
     * Pre-condition: `opr` is a 3-character string like "++X" or "X--".
     * Logic: If the character at index 1 is '+', it's an increment. If it's
     *        '-', it's a decrement. This works for all four variations of the
     *        operations.
     */
    if (opr.at(1) == '+')
      x++;
    if (opr.at(1) == '-')
      x--;
  }

  // Output the final value of x.
  cout << x << endl;

  return 0;
}
