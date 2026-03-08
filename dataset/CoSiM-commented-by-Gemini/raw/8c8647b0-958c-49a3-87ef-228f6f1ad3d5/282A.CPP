/**
 * @file 282A.CPP
 * @brief Solves the "Bit++" problem from Codeforces.
 *
 * @details
 * This program calculates the final value of a variable `x` after a series
 * of increment or decrement operations. The operations are given as strings
 * like "X++" or "--X". This solution uses an efficient method where it only
 * inspects the middle character of the operation string to determine whether
 * it's an increment or a decrement.
 *
 * @section
 * Algorithm: Iterative accumulation with middle-character check.
 * Time Complexity: O(N), where N is the number of operations.
 * Space Complexity: O(1) (ignoring the temporary string storage per loop).
 */
#include <bits/stdc++.h>
using namespace std;
int main() {
  // n: the number of operations.
  // x: the accumulator variable, starts at 0.
  int n, x = 0;
  // str: holds the operation string for each iteration.
  string str;
  cin >> n;
  /**
   * @brief Main loop to process each of the 'n' operations.
   */
  for (int i = 0; i < n; i++) {
    cin >> str;
    /**
     * @brief Determines the operation type by inspecting the middle character.
     *
     * Logic: For all valid inputs ("++X", "X++", "--X", "X--"), the character
     * at index 1 is sufficient to identify the operation. This avoids the need
     * for full string comparisons.
     */
    if (str[1] == '+') {
      x++;
    }
    // Using a separate 'if' is slightly less optimal than 'else if', but works
    // correctly here as the conditions are mutually exclusive.
    if (str[1] == '-') {
      x--;
    }
  }
  // Output the final value of x.
  cout << x;
  return 0;
}
