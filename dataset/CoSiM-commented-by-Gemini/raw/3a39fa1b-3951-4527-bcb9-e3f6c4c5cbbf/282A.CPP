/**
 * @file 282A.CPP
 * @brief Solution to the "Bit++" problem (Codeforces 282A).
 *
 * @details The problem involves a simple programming language with one variable, X,
 * and two operations: `++` which increases X by 1, and `--` which decreases
 * X by 1. The program calculates the final value of X after a sequence of
 * these operations.
 *
 * @algorithm The program initializes a variable `value` to 0. It then loops `n`
 * times, reading one statement per iteration. The statements can be "X++",
 * "++X", "X--", or "--X". By checking only the first and third characters of
 * the input string (e.g., `statement[0]` and `statement[2]`), the program can
 * efficiently determine whether the operation is an increment or a decrement,
 * as the middle character is always 'X'. The final value is printed after
 * all statements are processed.
 *
 * @complexity
 * Time Complexity: O(N), where N is the number of statements, due to a single pass.
 * Space Complexity: O(1), as it uses a small, fixed-size buffer for the input
 *                   and a few integer variables.
 */
#include <iostream>

using std::cin;
using std::cout;

int main(void) {
  int n;

  cin >> n;

  int count = 0;
  int value = 0;
  /**
   * @brief Block Logic: Processes `n` statements from the input.
   * @invariant `value` holds the result of the first `count` operations.
   */
  while (count++ < n) {
    char statement[] = {0, 0, 0, 0}; // Buffer to read the 3-char statement.
    cin >> statement;
    // Check for "++X" or "X++".
    if (statement[0] == '+' || statement[2] == '+') {
      value++;
    } 
    // Check for "--X" or "X--".
    else if (statement[0] == '-' || statement[2] == '-') {
      value--;
    }
  }
  cout << value;

  return 0;
}