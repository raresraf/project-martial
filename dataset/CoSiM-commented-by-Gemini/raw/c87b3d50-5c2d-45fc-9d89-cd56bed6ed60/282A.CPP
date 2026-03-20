/**
 * @file 282A.CPP
 * @brief Solution for Codeforces problem 282A, "Bit++".
 * @details This program calculates the final value of a variable `y` after a
 * series of increment and decrement operations. This implementation uses
 * full string comparison to identify the operations.
 *
 * @section Algorithm
 * The program initializes a counter `y` to 0. It reads the total number of
 * operations into a variable `x`. It then declares a variable-length array (VLA)
 * of strings, `c[x]`, to store all operation strings. This VLA feature is a
 * non-standard C++ extension.
 *
 * The main loop iterates `x` times, reading each operation string into the array.
 * Inside the loop, a chain of `if-else if` statements compares the current
 * operation string against the four possibilities ("X++", "++X", "X--", "--X")
 * and updates the counter `y` accordingly.
 * Finally, the program prints the final value of `y`.
 *
 * @subsection Complexity
 * - Time Complexity: O(n), where `n` is the number of operations, due to the loop.
 * - Space Complexity: O(n), because it stores all `n` operation strings in an array.
 */
#include <bits/stdc++.h>
#include <string>
#define ll long long
using namespace std;
int main() {

/**
 * @brief This preprocessor block redirects standard I/O to files for local testing.
 * It is a common convenience in competitive programming for debugging. When the
 * code is compiled without the `ONLINE_JUDGE` macro defined, `stdin` will read
 * from "input.txt" and `stdout` will write to "output.txt".
 */
#ifndef ONLINE_JUDGE
  freopen("input.txt", "r", stdin);
  freopen("output.txt", "w", stdout);
#endif
  // x: stores the number of operations.
  // y: the counter, initialized to 0.
  int x, y = 0;
  cin >> x;
  // c: a variable-length array (VLA) to store the operation strings.
  // Note: VLAs are a non-standard extension in C++.
  string c[x];

  /**
   * @brief This loop reads all operation strings and updates the counter.
   * Pre-condition: `y` is 0, `x` is the number of operations.
   * Invariant: At the end of iteration `i`, `y` reflects the cumulative result
   * of the first `i+1` operations.
   */
  for (int i = 0; i < x; i++) {
    cin >> c[i];

    /**
     * @brief This block uses full string comparison to identify the operation
     * and update the counter `y`. It handles both prefix and postfix forms.
     */
    if (c[i] == "X++") {
      y++;
    } else if (c[i] == "++X") {
      ++y;
    } else if (c[i] == "X--") {
      y--;
    } else if (c[i] == "--X") {
      --y;
    }
  }
  // Print the final value of the counter.
  cout << y << endl;
}