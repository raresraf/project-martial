/**
 * @file 282A.CPP
 * @brief Solution for the "Bit++" problem from Codeforces.
 *
 * This problem involves a single variable, `x`, initialized to 0.
 * A series of `n` operations are performed on `x`. Each operation is given
 * as a string, which can be either "++X", "X++", "--X", or "X--".
 * The task is to determine the final value of `x` after all operations.
 *
 * Problem Link: https://codeforces.com/problemset/problem/282/A
 *
 * Algorithm: Iterative processing of string operations, accumulating the change in a variable.
 * Time Complexity: O(N), where N is the number of operations. Each operation
 *                  string is of constant length (3 characters), so string
 *                  access and comparison is O(1).
 * Space Complexity: O(1), as only a few variables are used to store the count
 *                   and the current operation string.
 */

#include <iostream>
#include <string>
using namespace std;

int main() {
  int n(0), x(0); // n: number of operations, x: the variable to be manipulated (initialized to 0).
  // Reads the number of operations `n` from standard input.
  cin >> n;
  string s; // String to store each operation.

  /**
   * Block Logic: Processes `n` operations, incrementing or decrementing `x` based on the operation string.
   * Invariant: After each iteration, `x` holds the accumulated value after the processed operation.
   */
  for (int i = 0; i < n; i++) {
    // Reads the operation string (e.g., "++X", "X--").
    cin >> s;
    // The key character for determining the operation is at index 1 ('+' or '-').
    if (s[1] == '+') {
      x++; // Increment `x` if the operation is "++X" or "X++".
    } else {
      x--; // Decrement `x` if the operation is "--X" or "X--".
    }
  }
  // Prints the final value of `x`.
  cout << x << endl;
  return 0;
}
