/**
 * @file 282A.CPP
 * @brief A highly concise solution for Codeforces problem 282A, "Bit++".
 * @details This program calculates the final value of a variable `x` after
 * being modified by a series of increment or decrement operations.
 *
 * @section Algorithm
 * This implementation employs a very efficient and clever simplification to
 * determine the nature of each operation.
 *
 * 1. An integer `n` is read, representing the number of operations.
 * 2. A counter `x` is initialized to 0.
 * 3. The program loops `n` times. In each iteration, an operation string `s` is read.
 * 4. The core insight is that for all valid operations in this problem
 *    ("++X", "X++", "--X", "X--"), the character at index 1 is unique to the
 *    operation type. It is '+' for increments and '-' for decrements.
 * 5. The program checks `s[1]`. If it is '+', `x` is incremented; otherwise,
 *    it's decremented.
 * 6. After the loop, the final value of `x` is printed.
 *
 * @subsection Complexity
 * - Time Complexity: O(n), where `n` is the number of operations, due to the
 *   single loop processing each operation.
 * - Space Complexity: O(1), as the memory used for the operation string `s` is
 *   reused and does not scale with `n`.
 */
#include <bits/stdc++.h>
using namespace std;

int main() {
  // n: the number of operations.
  // x: the variable to be modified, initialized to 0.
  // s: stores the operation string.
  int n, x;
  string s;
  scanf("%d", &n);
  x = 0;
  /**
   * @brief This loop processes each of the `n` operations.
   * Pre-condition: `x` is 0, and `n` holds the total number of operations.
   * Invariant: After each iteration, `x` reflects the cumulative result of all
   * operations performed so far.
   */
  while (n--) {
    cin >> s;
    /**
     * @brief This conditional check is the core of the simplified logic.
     * The character at index 1 of the operation string is sufficient to
     * determine whether to increment or decrement the counter `x`.
     */
    if (s[1] == '+')
      x++;
    else
      x--;
  }
  printf("%d", x);
}
