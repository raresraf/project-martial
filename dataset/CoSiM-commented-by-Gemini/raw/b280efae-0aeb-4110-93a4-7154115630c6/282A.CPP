/**
 * @file 282A.CPP
 * @brief Solution to Codeforces problem 282A - "Bit++".
 * @details This program simulates a series of increment and decrement operations on a single variable, `x`, and prints its final value.
 * The logic checks for the presence of a '+' character in the operation string to determine whether to increment or decrement.
 *
 * Algorithm:
 * 1. Initialize a counter `x` to zero.
 * 2. Read the total number of operations, `n`.
 * 3. Loop `n` times:
 *    a. Read the operation string.
 *    b. If the string contains a '+', increment `x`.
 *    c. Otherwise (if it contains a '-'), decrement `x`.
 * 4. Print the final value of `x`.
 *
 * Time Complexity: O(N * L), where N is the number of operations and L is the length of the operation string (typically 3).
 *                  Since L is constant, complexity is effectively O(N).
 * Space Complexity: O(L) to store the operation string within the loop. Effectively O(1).
 */
// 282A.cpp
// A. Bit++
// Author: Ikki
// May 31, 2015

#include <iostream>
#include <string>
using namespace std;

int main() {
  int n, x = 0;
  cin >> n;
  /**
   * @brief Main loop to process `n` operations.
   * Invariant: `x` holds the cumulative value of the simulated variable after each operation.
   */
  for (int i = 0; i < n; i++) {
    string str;
    cin >> str;
    /**
     * @brief Checks the operation type by searching for a '+' character.
     * The `string::find` method returns `string::npos` if the character is not found.
     * This robustly distinguishes between increment and decrement operations.
     */
    if (str.find('+') != string::npos)
      x++;
    else
      x--;
  }
  // Output the final value of the variable 'x' after all operations.
  cout << x << endl;
  return 0;
}
