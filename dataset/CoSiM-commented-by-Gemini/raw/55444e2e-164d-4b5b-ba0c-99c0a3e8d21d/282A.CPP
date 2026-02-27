/**
 * @file 282A.CPP
 * @brief Solves the "Bit++" problem from Codeforces.
 * @details The problem involves a single variable, X, initialized to 0. The
 * program is given a sequence of operations, each being one of "++X", "X++",
 * "--X", or "X--", and must calculate the final value of X.
 *
 * @algorithm
 * The algorithm simulates the process directly.
 * 1. Initialize a variable `x` to 0.
 * 2. Read the number of operations, n.
 * 3. Loop n times:
 *    a. Read the operation as a string.
 *    b. Explicitly check if the string is either "++X" or "X++".
 *    c. If it is, increment `x`.
 *    d. Otherwise, the operation must be a decrement, so decrement `x`.
 * 4. After all operations, print the final value of `x`.
 *
 * @complexity
 * Time Complexity: O(n), for processing n operations.
 * Space Complexity: O(1), as the space used is constant regardless of n.
 */
#include <bits/stdc++.h>
using namespace std;

int main() {
  int n; // The number of operations.
  int x = 0; // The variable X, initialized to 0.
  string word; // To store the operation string.
  
  cin >> n;
  
  /**
   * @block
   * @brief Process each operation.
   */
  for (int i = 0; i < n; i++) {
    cin >> word;
    // Check if the operation is an increment.
    if (word == "++X" || word == "X++")
      x++;
    else // Otherwise, it's a decrement.
      x--;
  }

  cout << x << endl;

  return 0;
}
