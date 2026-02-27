/**
 * @file 282A.CPP
 * @brief Solves the "Bit++" problem from Codeforces.
 * @details The problem involves a variable `X` initialized to 0. A sequence of
 * operations is given, each being one of `++X`, `X++`, `--X`, or `X--`. The goal
 * is to find the final value of `X`.
 *
 * @algorithm
 * A simple and clever approach is to realize that any increment operation
 * contains the substring "++", and any decrement operation contains "--".
 * 1. Initialize a result variable `res` (representing X) to 0.
 * 2. Read the number of operations, n.
 * 3. Loop n times:
 *    a. Read the operation as a string.
 *    b. Check if the substring "++" exists within the operation string.
 *    c. If it exists, increment `res`.
 *    d. Otherwise, the operation must be a decrement, so decrement `res`.
 * 4. After the loop, print the final value of `res`.
 *
 * @complexity
 * Time Complexity: O(n), where n is the number of operations. Each string
 * operation (`find`) is on a short, constant-length string, making it O(1).
 * Space Complexity: O(1), as we only store one operation string at a time.
 */
#include <bits/stdc++.h>
using namespace std;

int main() {
  // Fast I/O for competitive programming.
  ios_base::sync_with_stdio(false);
  cin.tie(NULL);

  string in; // To store the operation string, e.g., "++X".
  int n;     // The number of operations.
  cin >> n;
  
  int res = 0; // Represents the variable X, initialized to 0.

  /**
   * @block
   * @brief Process each operation.
   */
  while (n--) {
    cin >> in;
    /**
     * @brief Check for the presence of "++" to determine the operation type.
     * `string::npos` is a constant indicating that the substring was not found.
     */
    if (in.find("++") != in.npos)
      res++; // Increment if "++" is found.
    else
      res--; // Decrement otherwise (it must be a "--" operation).
  }
  
  // Print the final value of X.
  cout << res;
}
