/**
 * @file 282A.CPP
 * @brief Solution for the "Bit++" programming contest problem.
 * @details The program simulates 'n' operations on a variable 'x', which starts
 * at 0. The operations are given as strings and are either an increment
 * ("++X" or "X++") or a decrement ("--X" or "X--"). This solution uses the
 * `std::string::find` method to determine the operation type.
 *
 * Algorithm:
 * 1. Initialize an accumulator variable `x` to 0.
 * 2. Read the number of operations, `n`.
 * 3. Loop 'n' times:
 *    a. Read the operation string.
 *    b. Use `string::find("++")` to check if the string contains the increment
 *       substring "++".
 *    c. If it does, increment `x`.
 *    d. Otherwise, assume it's a decrement operation and decrement `x`.
 * 4. After the loop, print the final value of `x`.
 *
 * Time Complexity: O(n), where 'n' is the number of operations.
 * Space Complexity: O(L_max), where L_max is the maximum length of an operation string (constant).
 */
#include <iostream>
#include <string>

using namespace std;

int main() {
  int x = 0;

  int n;
  cin >> n;

  string line;
  /**
   * @block
   * @description This loop processes each of the 'n' operations.
   */
  for (int i = 0; i < n; i++) {
    cin >> line;
    /**
     * @block
     * @description Determines the operation type by searching for the "++" substring.
     * If "++" is found, it's an increment; otherwise, it's a decrement.
     */
    if (line.find("++") != string::npos) {
      x++;
    } else {
      x--;
    }
  }

  // Output the final value of x.
  cout << x << endl;
}