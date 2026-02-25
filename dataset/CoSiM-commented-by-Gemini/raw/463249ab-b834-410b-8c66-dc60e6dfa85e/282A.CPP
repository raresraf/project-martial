/**
 * @file 282A.CPP
 * @brief Solution for a simple simulation problem from a programming contest.
 * @details The program simulates a series of increment and decrement operations on a
 * variable 'x', which starts at 0. It reads 'n' operations and calculates the
 * final value of 'x'.
 *
 * Algorithm:
 * 1. Initialize a variable 'x' to 0.
 * 2. Read the total number of operations, 'n'.
 * 3. Loop 'n' times:
 *    a. Read the operation string (e.g., "X++", "++X", "X--", "--X").
 *    b. If the string represents an increment operation, increment 'x'.
 *    c. If the string represents a decrement operation, decrement 'x'.
 * 4. After the loop, print the final value of 'x'.
 *
 * Time Complexity: O(n), where 'n' is the number of operations, as each operation
 * is processed in constant time.
 * Space Complexity: O(1), as the storage required is constant regardless of 'n'.
 */
#include <iostream>
#include <string>

using namespace std;

int main() {
  string s;
  int n, x = 0;
  cin >> n;
  
  /**
   * @block
   * @description This loop processes each of the 'n' operations.
   * Invariant: At the end of each iteration, 'x' holds the accumulated value
   * after processing the operations so far.
   */
  for (int i = 0; i < n; i++) {
    cin >> s;
    
    /**
     * @block
     * @description Checks the operation string and updates the value of 'x' accordingly.
     * "X++" and "++X" increment the value.
     * "X--" and "--X" decrement the value.
     */
    if (s == "X++" || s == "++X") {
      x++;
    } else if (s == "X--" || s == "--X") {
      x--;
    }
  }
  
  // Output the final value of x after all operations.
  cout << x << endl;
  return 0;
}
