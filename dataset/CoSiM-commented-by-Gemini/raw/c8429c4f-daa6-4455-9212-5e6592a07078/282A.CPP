/**
 * @file 282A.CPP
 * @brief Solution for Codeforces problem 282A, "Bit++".
 * @details This program evaluates a series of simple increment and decrement
 * operations on a single variable, `x`, and outputs its final value.
 *
 * @section Algorithm
 * The program initializes a variable `x` to 0. It then reads an integer `n`,
 * which specifies the total number of operations to follow. It loops `n` times,
 * reading one operation string in each iteration. The operation can be one of four types:
 * "++X", "X++", "--X", or "X--".
 *
 * The core logic involves checking the operation string to determine whether it's
 * an increment or a decrement. A simple way to do this, which the code implements
 * through a series of checks, is to identify the operator characters. If the operation
 * involves a '+', `x` is incremented. If it involves a '-', `x` is decremented.
 * After all operations are processed, the final value of `x` is printed.
 *
 * A more concise implementation could check the character at index 1 of the string,
 * as it is always '+' for an increment and '-' for a decrement.
 *
 * @subsection Complexity
 * - Time Complexity: O(n), where `n` is the number of operations, as each operation is processed once.
 * - Space Complexity: O(1), as the program uses a fixed amount of memory.
 */
#include <bits/stdc++.h>
using namespace std;
int main() {
  // n: the number of operations to perform.
  // x: the variable to be manipulated, initialized to 0.
  int n, x = 0;
  string operation;

  cin >> n;
  // This condition ensures `n` is within the problem's constraints.
  if (1 <= n && n <= 150) {
    /**
     * @brief This loop reads and processes each of the `n` operations.
     * Pre-condition: `x` is 0 and `n` is the number of operations.
     * Invariant: After each iteration, `x` holds the cumulative result of all
     * operations processed so far.
     */
    while (n--) {
      cin >> operation;

      /**
       * @brief This block of conditional statements identifies the type of operation
       * and updates `x` accordingly. It covers both prefix and postfix forms.
       */
      if (operation[0] == 'X') {
        if (operation[1] == '+' && operation[2] == '+') {
          x++;
        } else if (operation[1] == '-' && operation[2] == '-') {
          x--;
        }

      } else if (operation[0] == '+') {
        if (operation[1] == '+' && operation[2] == 'X') {
          x++;
        }

      } else if (operation[0] == '-') {
        if (operation[1] == '-' && operation[2] == 'X') {
          x--;
        }
      }
    }
  }
  // Print the final value of x after all operations have been performed.
  cout << x;
  return 0;
}
