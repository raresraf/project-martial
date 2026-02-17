/**
 * @file 282A.CPP
 * @brief Solution for Codeforces Problem 282A, "Bit++".
 * @details This program simulates a series of operations on a single variable, `x`,
 * which is initialized to 0. The operations are given as strings like "++X" or "X--".
 * The program calculates and prints the final value of `x`.
 *
 * @algorithm Iterative Counting via Character Inspection
 * @description The program reads the number of operations, then loops that many times.
 * In each iteration, it reads the operation as a string. It determines whether to
 * increment or decrement the variable `x` by checking the character at the second
 * position (index 1) of the string. If it's a '+', `x` is incremented; otherwise,
 * it's decremented.
 * @complexity
 * - Time: O(N), where N is the number of operations, because it processes each
 *   operation once.
 * - Space: O(1), as it uses a fixed amount of memory regardless of input size.
 */
#include <iostream>
#include <string>
using namespace std;
int main() {
  int n, x = 0;
  cin >> n;
  string sta;

  /**
   * @brief This loop processes each of the 'n' operations.
   * @invariant `x` holds the accumulated value after the first `i` operations.
   */
  while (n--) {
    cin >> sta;
    /**
     * @brief This condition efficiently determines the operation type.
     * For all possible inputs ("++X", "X++", "--X", "X--"), the character at
     * index 1 is unique to the operation type ('+' for increment, '-' for decrement).
     */
    if (sta[1] == '+')
      x++;
    else
      x--;
  }
  
  // Output the final value of x.
  cout << x << endl;
  return 0;
}
