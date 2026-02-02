/**
 * @file 282A_2.CPP
 * @brief Solution for Codeforces Problem 282A - Bit++.
 *
 * This problem involves a single variable `X` and a series of `N` operations.
 * Each operation is given as a string and can either increment `X` ("++X" or "X++")
 * or decrement `X` ("--X" or "X--"). The task is to calculate the final value of `X`
 * after all operations, starting with `X = 0`. This solution represents an alternative
 * parsing approach to problem 282A.
 *
 * Algorithm: Character-based string parsing and counter update.
 * The code initializes a counter `x` to 0. For each operation string, it checks specific
 * characters within the string to determine if it represents an increment or decrement
 * operation. If `X` is at index 0 and `+` is at index 1, or if `X` is at index 2 and `+`
 * is at index 0, it increments `x`. Otherwise, it decrements `x`.
 *
 * Time Complexity: O(N), where N is the number of operations, as each operation string
 * is processed in constant time (fixed length string checks).
 * Space Complexity: O(L) for storing the operation string, where L is its fixed length (3).
 */

#include <bits/stdc++.h> // Includes most standard C++ libraries, common in competitive programming.
using namespace std; // Functional Utility: Brings all identifiers from the std namespace into the current scope.

/**
 * @brief Main function: Solves the Bit++ problem.
 * Functional Utility: Reads a number of operations `n`, processes each operation
 * (increment or decrement of `X`), and prints the final value of `X`.
 * Algorithm: String parsing to identify increment/decrement and update a counter.
 */
int main() {
  int n; // Declares an integer `n` to store the number of operations.
  int x = 0; // Initializes the variable `x` (representing `X`) to 0.
  string operation; // Declares a string `operation` to store the current operation.
  
  cin >> n; // Reads the number of operations `n`.
  
  // Block Logic: Loop `n` times to process each operation.
  // Precondition: `n` is a non-negative integer.
  // Invariant: `x` reflects the sum of operations up to the current iteration.
  while (n--) {
    cin >> operation; // Reads the current operation string.
    
    // Block Logic: Determine if the operation is an increment or decrement.
    // This logic checks two main patterns: 'X++'/'X--' or '++X'/`--X`.
    if (operation[0] == 'X') { // If the character 'X' is at the first position (e.g., "X++", "X--").
      if (operation[1] == '+') { // If the second character is '+'.
        x++; // Increment `x`.
      } else { // Otherwise (implies '-').
        x--; // Decrement `x`.
      }
    } else if (operation[2] == 'X') { // If the character 'X' is at the third position (e.g., "++X", "--X").
      if (operation[0] == '+') { // If the first character is '+'.
        x++; // Increment `x`.
      } else { // Otherwise (implies '-').
        x--; // Decrement `x`.
      }
    }
  }
  cout << x; // Prints the final calculated value of `x`.

  return 0; // Indicate successful program execution.
}