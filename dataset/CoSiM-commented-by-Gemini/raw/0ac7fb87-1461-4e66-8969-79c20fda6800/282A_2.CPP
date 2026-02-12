/**
 * @file 282A_2.CPP
 * @brief Solves the "Bit++" problem by inspecting characters of the operation string.
 * @details This program calculates the final value of a variable `X` initialized to 0. It reads
 * an integer `n`, the number of operations, and then loops `n` times. In each iteration, it reads
 * an operation string (e.g., "++X", "X--"). Instead of comparing the whole string, it checks
 * specific character positions to determine if the operation is an increment or a decrement.
 *
 * @algorithm Character-based parsing of operation strings.
 * @complexity
 * Time complexity: O(N), where N is the number of operations, as each string is processed in constant time.
 * Space complexity: O(1), aside from storing the operation string.
 */
#include <bits/stdc++.h>
using namespace std;
int main() {
  // n: the number of operations to perform.
  // x: the variable, initialized to 0.
  int n, x = 0;
  string operation;
  cin >> n;
  /**
   * @brief Main loop to process each operation.
   * @invariant Iterates `n` times, reading one operation and updating `x` in each iteration.
   */
  while (n--) {
    cin >> operation;
    /**
     * @brief Logic to determine whether to increment or decrement `x`.
     * @pre An operation string has been read.
     * @post `x` is updated based on the operation.
     * @details This block checks characters at specific positions to identify the operation,
     * which is an alternative to full string comparison.
     */
    if (operation[0] == 'X') {
      // Handles "X++" and "X--"
      if (operation[1] == '+') {
        x++;
      } else {
        x--;
      }
    } else if (operation[2] == 'X') {
      // Handles "++X" and "--X"
      if (operation[0] == '+') {
        x++;
      } else {
        x--;
      }
    }
  }
  // Print the final value of x.
  cout << x;

  return 0;
}