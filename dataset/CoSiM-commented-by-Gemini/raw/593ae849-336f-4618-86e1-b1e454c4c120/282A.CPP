/**
 * @file 282A.CPP
 * @brief Solves the "Bit++" problem from Codeforces.
 * @details This program calculates the final value of a variable 'X' that starts at 0
 * and is subjected to a series of increment and decrement operations.
 *
 * @algorithm
 * The solution initializes a counter `x` to 0. It reads the total number of
 * operations, `n`. It then enters a loop that iterates `n` times. In each iteration,
 * it reads an operation as a standard C++ string. It uses direct string comparison
 * to check if the operation is one of the two increment forms ("++X", "X++") or
 * one of the two decrement forms ("--X", "X--"). The counter `x` is updated
 * accordingly. After the loop completes, the final value of `x` is printed.
 *
 * @complexity
 * Time Complexity: O(N), where N is the number of operations. The program loops N times,
 * and each string comparison takes constant time for the short, fixed-length strings
 * in this problem.
 * Space Complexity: O(1), ignoring the storage for the input string within the loop,
 * as it uses a fixed number of variables.
 */
#include <bits/stdc++.h>
using namespace std;

int main() {
  /**
   * @variable n: The total number of operations to perform.
   */
  int n;

  cin >> n;

  /**
   * @variable x: The variable being operated on, initialized to 0.
   */
  int x = 0;
  /**
   * @block
   * @brief This loop processes each of the `n` operations.
   * @invariant `x` holds the accumulated result of the first `i-1` operations.
   */
  for (int i = 1; i <= n; i++) {
    /**
     * @variable statement: The string representing the current operation (e.g., "++X").
     */
    string statement;
    cin >> statement;

    // Pre-condition: Check if the operation is an increment.
    if (statement == "++X" || statement == "X++") {
      x += 1;
    // Pre-condition: Check if the operation is a decrement.
    } else if (statement == "--X" || statement == "X--") {
      x -= 1;
    }
  }

  // Output the final value of x.
  cout << x << endl;

  return 0;
}
