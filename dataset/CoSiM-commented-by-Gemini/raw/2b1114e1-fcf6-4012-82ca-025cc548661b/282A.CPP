/**
 * @file 282A.CPP
 * @brief Solution to the "Bit++" problem (282A) from Codeforces.
 * @details The program calculates the final value of a variable 'X' after a series of
 * increment or decrement operations. The operations can be "++X", "X++", "--X", or "X--".
 *
 * @author Your Name (or leave blank)
 *
 * @algorithm This clever solution determines the operation type (increment or decrement)
 * by inspecting only the middle character of the operation string.
 * 1. Read the number of statements, `n`.
 * 2. Initialize a variable `x` to 0.
 * 3. Loop `n` times to process each statement.
 * 4. In each iteration, read the statement as a string.
 * 5. Check the middle character of the string (`str[1]`). For all increment operations
 *    ("++X" and "X++"), the middle character is '+'. For all decrement operations, it is '-'.
 * 6. If the middle character is '+', increment `x`.
 * 7. Otherwise, decrement `x`.
 * 8. After the loop, print the final value of `x`.
 *
 * @complexity
 * Time Complexity: O(N), where N is the number of statements, due to the single loop.
 * Space Complexity: O(1), as it only stores one statement string at a time.
 */
#include <bits/stdc++.h>
using namespace std;

int main() {
  int n;
  cin >> n;

  int x = 0;

  /**
   * @block
   * @brief Processes `n` statements, updating the value of `x`.
   */
  while (n--) {
    string str;
    cin >> str;

    /**
     * @block
     * @brief Determines the operation by checking the middle character of the statement string.
     * @details The ASCII value 43 corresponds to the '+' character. If the middle char
     *          is '+', it's an increment; otherwise, it's a decrement.
     */
    if ((int)str[1] == 43) {
      x++;
    } else {
      x--;
    }
  }
  cout << x << endl;

  return 0;
}
