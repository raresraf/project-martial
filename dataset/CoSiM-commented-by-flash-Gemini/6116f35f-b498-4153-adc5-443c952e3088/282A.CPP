/**
 * @file 282A.CPP
 * @brief Solution for Codeforces Problem 282A - "Bit++".
 * This program simulates operations on a single integer variable 'x',
 * which starts at 0. Operations are either increment ('++X', 'X++')
 * or decrement ('--X', 'X--'). The goal is to find the final value of 'x'.
 * Algorithm: Iteratively apply operations to a counter.
 * Time Complexity: O(N), where N is the number of operations. String comparisons
 *                  and arithmetic operations take constant time per iteration.
 * Space Complexity: O(1) - uses a constant amount of memory for variables.
 */
#include <bits/stdc++.h> // Includes most standard C++ libraries
using namespace std;     // Uses the standard namespace to avoid prefixing std::

int main() {
  // n: stores the total number of operations.
  // x: the variable being manipulated, initialized to 0.
  int n, x = 0;

  // Reads the total number of operations from standard input.
  cin >> n;

  string s; // Declares a string to store each operation.

  /**
   * Block Logic: Processes each operation one by one, updating the value of 'x'.
   * The loop runs 'n' times, decrementing 'n' in each iteration.
   * Invariant: `x` holds the current value after applying all operations processed so far.
   */
  while (n--) {
    cin >> s; // Reads the operation string.
    // Checks if the operation is an increment.
    // Both "++X" and "X++" result in an increment.
    if (s == "++X" || s == "X++")
      x++; // Increments x.
    else
      x--; // Otherwise, it's a decrement operation, so decrement x.
  }
  
  // Prints the final value of 'x' after all operations.
  cout << x << endl;

  return 0; // Indicates successful program execution.
}
