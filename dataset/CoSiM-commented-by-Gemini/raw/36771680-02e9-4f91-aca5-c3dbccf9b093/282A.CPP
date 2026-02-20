/**
 * @file 282A.CPP
 * @brief Solves the "Bit++" programming language problem.
 *
 * This program simulates a simple programming language where a single variable 'X'
 * is initialized to 0. It then executes a series of statements, which are either
 * increment (`++X`, `X++`) or decrement (`--X`, `X--`) operations. The final
 * value of X is printed.
 *
 * Algorithm:
 * 1. Read the number of statements, 'a'.
 * 2. Initialize a variable 'x' to 0.
 * 3. Loop 'a' times:
 *    a. Read the statement string.
 *    b. If the string contains "++", increment 'x'.
 *    c. If the string contains "--", decrement 'x'.
 * 4. Print the final value of 'x'.
 *
 * Time Complexity: O(N), where N is the number of statements.
 * Space Complexity: O(1), ignoring the storage for the input string which is reused.
 */
#include <bits/stdc++.h>
using namespace std;
int main() {
  string ch; // To store the statement string e.g., "++X".
  int a, i, x; // a: number of statements, i: loop counter, x: the variable.

  /**
   * Block Logic: Outer loop for multiple test cases.
   * This `while` loop structure allows the program to solve multiple sets of
   * operations if the input provides them, though typically for this problem,
   * only one set is given.
   */
  while (cin >> a) {
    x = 0; // Initialize x to 0 for each new set of operations.
    
    /**
     * Block Logic: Main processing loop for a single test case.
     * Iterates through 'a' statements.
     */
    for (i = 1; i <= a; i++) {
      cin >> ch;
      // Pre-condition: Check if the statement is an increment.
      if (ch == "++X" || ch == "X++") {
        x++;
      } 
      // Pre-condition: Check if the statement is a decrement.
      else if (ch == "--X" || ch == "X--") {
        x--;
      }
    }
    // Print the final value of x after all operations.
    cout << x << endl;
  }
  return 0;
}
