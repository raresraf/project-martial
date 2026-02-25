/**
 * @file 282A.CPP
 * @brief Solution for the "Bit++" programming contest problem.
 * @details The program simulates 'n' increment or decrement operations on a
 * variable 'x', which starts at 0. This implementation uses the
 * `std::string::compare` method for identifying the operations.
 *
 * Algorithm:
 * 1. Initialize an accumulator variable `x` to 0.
 * 2. Read the number of operations, `n`.
 * 3. Loop 'n' times:
 *    a. Read the operation string.
 *    b. Use `string::compare()` to check if the string matches "++X" or "X++".
 *       If it matches, increment `x`.
 *    c. Else, check if the string matches "--X" or "X--". If it matches,
 *       decrement `x`.
 * 4. After the loop, print the final value of `x`.
 *
 * Time Complexity: O(n), where 'n' is the number of operations.
 * Space Complexity: O(L_max), where L_max is the maximum length of an operation string (constant).
 */
#include <bits/stdc++.h>

using namespace std;

int main() {
  int n;
  int x = 0;
  string command;
  cin >> n;
  
  /**
   * @block
   * @description This loop processes each of the 'n' operations.
   */
  for (int i = 0; i < n; ++i) {
    cin >> command;
    /**
     * @block
     * @description Explicitly compares the input string with the four possible
     * operation strings to determine whether to increment or decrement x.
     */
    if (command.compare("++X") == 0 || command.compare("X++") == 0) {
      x++;
    } else if (command.compare("--X") == 0 || command.compare("X--") == 0) {
      x--;
    }
  }
  
  // Output the final value of x.
  cout << x << endl;
}

// Build command: g++ -std=c++11 -O2 -Wall 282A.cpp -o out
