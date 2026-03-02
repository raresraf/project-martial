#include <iostream>

using namespace std;

/**
 * @file 282A.CPP
 * @brief Solution to the "Bit++" problem from Codeforces (Problem 282A).
 * @details This program calculates the final value of a variable `X` after being
 * modified by a series of increment or decrement operations.
 *
 * Algorithm:
 * The program initializes a variable `X` to 0. It reads an integer `n` which
 * specifies the number of operations to follow. It then loops `n` times. In each
 * loop, it reads an operation as a string. It compares this string to the two
 * possible increment operations ("++X" and "X++"). If it matches, `X` is incremented.
 * Otherwise, it's assumed to be a decrement operation, and `X` is decremented.
 * After all operations have been processed, the final value of `X` is printed.
 *
 * Time Complexity: O(n), as each of the `n` operations is processed in constant time
 * (string comparison is on very short, fixed-length strings).
 * Space Complexity: O(1), as memory usage is constant.
 */
int main() {

  string str, s1 = "++X", s2 = "X++";
  int n, X = 0;
  cin >> n;
  
  /**
   * @brief Loop through each of the `n` operations.
   */
  for (int i = 0; i < n; i++) {
    cin >> str;
    /**
     * @brief Check the operation string to determine whether to increment or decrement.
     */
    if (str == s1 || str == s2) {
      X++;
    } else {
      X--;
    }
  }

  cout << X << endl;

  return 0;
}