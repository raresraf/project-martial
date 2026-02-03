/**
 * @file 282A.CPP
 * @brief Solution to the "Bit++" problem.
 * @see https://codeforces.com/problemset/problem/282/A
 *
 * This program simulates a simple programming language with one variable `x`
 * and two operations: `++x` and `--x`. It calculates the final value of `x`
 * after a series of operations.
 *
 * Algorithm:
 * 1. Initialize a variable `x` to 0.
 * 2. Read the number of operations (n).
 * 3. Loop n times:
 *    a. Read the operation as a string.
 *    b. If the second character of the string is '+', increment `x`.
 *    c. If the second character is '-', decrement `x`.
 * 4. Print the final value of `x`.
 *
 * Time Complexity: O(N), where N is the number of operations.
 * Space Complexity: O(1), as only a few variables are used regardless of input size.
 */
#include <iostream>
using namespace std;

int main() {
  int n, x = 0;
  char c[30];
  cin >> n;
  // Invariant: The loop processes n operations, updating the value of x
  // accordingly.
  for (int i = 0; i < n; i++) {
    cin >> c;
    // Block Logic: Checks the second character of the operation string to
    // determine whether to increment or decrement x. This works because both
    // "++X" and "X++" have '+' at index 1, and similarly for '-'.
    if (c[1] == '+')
      ++x;
    else if (c[1] == '-')
      --x;
  }
  cout << x;
  return 0;
}