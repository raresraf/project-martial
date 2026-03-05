/**
 * @file 282A.CPP
 * @brief Solves the "Bit++" problem from competitive programming platforms.
 *
 * This program calculates the final value of a variable `X` after a series of
 * operations. The variable `X` starts at 0. The operations are given as strings
 * like "++X", "X++", "--X", and "X--". Each operation either increments or
 * decrements `X` by one.
 *
 * @algorithm
 * The core idea is to determine whether an operation is an increment or a decrement.
 * Any operation string containing a '+' character increases the value, and any
 * string containing a '-' character decreases it.
 *
 * 1. Initialize a counter `c` to 0, representing the initial value of `X`.
 * 2. Read an integer `n`, the number of operations to follow.
 * 3. Loop `n` times to process each operation.
 *    a. Read the operation string `s`.
 *    b. Check if the string `s` contains the '+' character. The `string::find`
 *       method returns `string::npos` if the character is not found.
 *    c. If a '+' is found, it's an increment operation, so add 1 to `c`.
 *    d. Otherwise, it must be a decrement operation, so subtract 1 from `c`.
 * 4. After the loop has processed all `n` operations, print the final value of `c`.
 *
 * @complexity
 * Time Complexity: O(n * L), where `n` is the number of operations and `L` is the
 * length of the operation string (which is constant at 3). The `find` operation
 * takes time proportional to the string length.
 * Space Complexity: O(L) to store the operation string in each iteration.
 */
#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdio.h>
#include <string>
#include <vector>
using namespace std;

// The following are common competitive programming macros, but are unused in this file.
typedef vector<int> vi;
#define all(c) (c).begin(), (c).end()
#define pb push_back
#define tr(a)                                                                  \
  cout << "Vector: ";                                                          \
  for (auto i : a)                                                             \
    cout << i << " ";                                                          \
  cout << "\n";

int main() {
  // Fast I/O operations.
  ios_base::sync_with_stdio(false);
  cin.tie(NULL);

  // n: number of statements/operations.
  // c: the counter, representing the variable X, initialized to 0.
  int n, i, c = 0;
  cin >> n;

  // s: stores the operation string.
  string s;

  /**
   * @brief Loop to process each operation.
   */
  for (i = 0; i < n; i++) {
    cin >> s;
    // Ternary operator to check for increment or decrement.
    // If `s.find('+')` does not return `s.npos`, it means '+' was found.
    c += (s.find('+') != s.npos) ? 1 : -1;
  }

  // Print the final value of the counter.
  cout << c;
  return 0;
}
