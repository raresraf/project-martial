/**
 * @file 282A.CPP
 * @brief Solves the "Bit++" problem from Codeforces.
 * @details The problem involves a single variable, X, initialized to 0, and a
 * series of operations ("++X", "X++", "--X", "X--"). The goal is to find the
 * final value of X.
 *
 * @algorithm
 * This solution uses a clever observation to simplify the logic. In all given
 * operation strings, the character at index 1 is unique to the operation type:
 * it is '+' for increments and '-' for decrements.
 * 1. Initialize a variable `x` (representing X) to 0.
 * 2. Read the number of operations, n.
 * 3. Loop n times:
 *    a. Read the operation as a string `s`.
 *    b. Check the character at index 1 (`s[1]`).
 *    c. If `s[1]` is '+', increment `x`.
 *    d. Otherwise, decrement `x`.
 * 4. After the loop, print the final value of `x`.
 *
 * @complexity
 * Time Complexity: O(n), for processing n operations.
 * Space Complexity: O(1), as the space used is constant regardless of n.
 */
#include <iostream>
using namespace std;

int main() {
  int n, x(0);
  cin >> n;

  string s;
  while (n--) {
    cin >> s;
    /**
     * @brief Determine operation type by checking the character at index 1.
     * For "++X" and "X++", s[1] is '+'. For "--X" and "X--", it is '-'.
     * This is sufficient to distinguish between increment and decrement.
     */
    if (s[1] == '+') {
      x++;
    } else {
      x--;
    }
  }

  cout << x;
}
