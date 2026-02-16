/**
 * @file 282A.CPP
 * @brief Solution for Codeforces problem 282A, "Bit++".
 *
 * @description
 * This program calculates the final value of a variable `X` after a series of
 * operations. The initial value of `X` is 0. The operations are given as
 * strings: "++X", "X++", "--X", and "X--". The first two increment X, and the
 * last two decrement X.
 *
 * @algorithm
 * 1. Initialize a variable `x` to 0.
 * 2. Read the number of statements, `n`.
 * 3. Loop `n` times:
 *    a. Read the statement string.
 *    b. Check if the string corresponds to an increment operation ("X++" or "++X").
 *    c. If it is an increment, increment `x`.
 *    d. Otherwise, it must be a decrement, so decrement `x`.
 * 4. After the loop, print the final value of `x`.
 *
 * @time_complexity O(N), where N is the number of statements, as each statement
 *   is processed once.
 * @space_complexity O(1), ignoring the string storage which is reused.
 */
#include <iostream>
using namespace std;
int main() {
  // n: the number of statements to execute.
  // x: the variable, initialized to 0.
  int n, x = 0;
  cin >> n;

  // Block Logic: Process each of the 'n' statements.
  while (n--) {
    string s;
    cin >> s;

    // Block Logic: Check if the statement is an increment or decrement.
    // An alternative check could be `if (s[1] == '+')`, which covers both
    // "X++" and "++X".
    if (s == "X++" || s == "++X") {
      x++;
    } else {
      x--;
    }
  }
  cout << x;
  return 0;
}
