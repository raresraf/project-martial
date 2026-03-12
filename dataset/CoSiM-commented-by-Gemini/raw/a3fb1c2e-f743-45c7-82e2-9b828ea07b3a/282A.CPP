/**
 * @file 282A.CPP
 * @brief Solution to the "Bit++" problem from Codeforces.
 *
 * This program simulates a series of simple increment and decrement operations
 * on a single variable, `X`, and outputs its final value. The operations are
 * given as strings: "++X", "X++", "--X", and "X--".
 *
 * @algorithm Linear Scan/Simulation
 * @time_complexity O(N), where N is the number of operations, as it processes each operation once.
 * @space_complexity O(1), ignoring the storage for the input string which is reused.
 */
#include <cstring>
#include <iostream>
using namespace std;
int main() {
  string s;
  int i, n, j, k, l; // j will represent the variable X. Other variables are unused.
  cin >> n;
  j = 0; // Initialize X to 0.

  /**
   * @brief Reads and processes each operation.
   *
   * The loop iterates `n` times. In each iteration, it reads an operation string
   * and checks its format to determine whether to increment or decrement `j`.
   * The non-exclusive `if` statements work correctly because the patterns for
   * the four possible operations are distinct.
   */
  for (i = 0; i < n; i++) {
    cin >> s;
    // Corresponds to "X++"
    if (s[0] == 'X' && s[1] == '+') {
      j++;
    }
    // Corresponds to "++X"
    if (s[0] == '+') {
      j++;
    }
    // Corresponds to "X--"
    if (s[0] == 'X' && s[1] == '-') {
      j--;
    }
    // Corresponds to "--X"
    if (s[0] == '-') {
      j--;
    }
  }
  cout << j << endl;
  return 0;
}