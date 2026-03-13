/**
 * @file 282A.CPP
 * @brief Solution for CodeForces Problem 282A - "Bit++".
 * 
 * @details
 * This program simulates a series of operations on a variable `x`, which is initialized to 0.
 * The operations are given as strings: "++X", "X++", "--X", or "X--".
 * The program reads `n` such operations and calculates the final value of `x`.
 *
 * @algorithm
 * 1. Initialize `x` to 0.
 * 2. Read the number of operations, `n`.
 * 3. Loop `n` times:
 *    a. Read the operation string `s`.
 *    b. Check the second character of the string (`s[1]`):
 *       - If it's a '+', increment `x`.
 *       - If it's a '-', decrement `x`.
 * 4. Print the final value of `x`.
 *
 * @time_complexity O(N), where N is the number of operations, as each operation is processed in constant time.
 * @space_complexity O(1), ignoring the storage for the input string which is reused.
 */
#include <bits/stdc++.h>
using namespace std;

int main() {
  int n;      // Number of operations.
  int x = 0; // The variable to be manipulated, initialized to 0.
  cin >> n;
  string s; // Stores the operation string (e.g., "++X").

  /**
   * @brief Main loop to process each operation.
   * Invariant: `x` holds the accumulated value from the operations processed so far.
   */
  while (n--) {
    cin >> s;
    /**
     * @brief Logic to determine whether to increment or decrement.
     * This relies on the fact that for both "++X" and "X++", the second character is '+'.
     * For "--X" and "X--", the second character is '-'.
     */
    if (s[1] == '+') {
      ++x;
    } else {
      --x;
    }
  }

  // Output the final value of x.
  cout << x << endl;
  return 0;
}
