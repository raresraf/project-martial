/**
 * @file 282A.CPP
 * @brief Solves the "Bit++" problem from Codeforces.
 * @details The problem is to calculate the final value of a variable `X`,
 * initialized to 0, after a series of operations. The operations can be
 * one of "++X", "X++", "--X", or "X--".
 *
 * @algorithm
 * The algorithm directly simulates the operations as described.
 * 1. Initialize a variable `x` to 0.
 * 2. Read the number of operations, `n`.
 * 3. Loop `n` times:
 *    a. Read the operation as a string.
 *    b. If the string is "++X" or "X++", increment `x`.
 *    c. If the string is "--X" or "X--", decrement `x`.
 * 4. After all operations, print the final value of `x`.
 *
 * @complexity
 * Time Complexity: O(n), for processing `n` operations.
 * Space Complexity: O(1), as the space used for the operation string is constant.
 */
#include <iostream>
using namespace std;

int main() {
  int n;
  int x = 0; // The variable to be incremented/decremented.
  int ans = 0; // This variable is declared but never used.
  
  cin >> n;
  string o; // To store the operation string.
  
  /**
   * @block
   * @brief Process each of the `n` operations.
   */
  for (int i = 0; i < n; i++) {
    cin >> o;
    // Explicitly check for the two increment operations.
    if (o == "X++" || o == "++X")
      x++;
    // Explicitly check for the two decrement operations.
    else if (o == "X--" || o == "--X")
      x--;
  }
  
  cout << x;
}
