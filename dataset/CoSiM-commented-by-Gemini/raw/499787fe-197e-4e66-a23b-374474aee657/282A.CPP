/**
 * @file 282A.CPP
 * @brief Solution to Codeforces problem 282A (Bit++).
 * @author Solved by: [User who submitted the code]
 *
 * @details
 * This program simulates a series of increment and decrement operations on a
 * single variable, `x`, which is initialized to zero. The operations are given
 * as strings ("++X", "X++", "--X", "X--"). The final value of `x` after all
 * operations is printed.
 *
 * @section algorithm
 * The algorithm iterates through a given number of operations (`n`). For each
 * operation, it reads a string and checks for the presence of either a '+' or
 * a '-' character.
 * - If a '+' is found, a global counter `x` is incremented.
 * - If a '-' is found, the counter `x` is decremented.
 * The loop processing the string is terminated immediately after the first
 * operator character is found, ensuring each string corresponds to a single
 * operation.
 *
 * @section complexity
 * - Time Complexity: O(N * L), where N is the number of operations and L is
 *   the length of the operation string. Since L is constant and small (always 3),
 *   the complexity simplifies to O(N).
 * - Space Complexity: O(L) or O(1) as it only needs to store one operation
 *   string at a time.
 */

#include <bits/stdc++.h>
using namespace std;

int main() {
  // n: the total number of operations to perform.
  // x: the variable, initialized to 0, to be modified by the operations.
  int n, x = 0;
  
  // Read the number of operations.
  cin >> n;

  /**
   * @brief Main loop to process each operation.
   *
   * Invariant: Before each iteration, `x` holds the cumulative result of all
   * previously processed operations.
   */
  while (n--) {
    string s;
    // Read the operation string (e.g., "++X", "X--").
    cin >> s;
    
    int l = s.length();
    /**
     * @brief Scans the operation string for an operator character.
     *
     * This loop finds the first operator character ('+' or '-') to determine
     * the nature of the operation. It breaks immediately after finding one,
     * effectively treating the entire string as a single increment or decrement.
     */
    for (int i = 0; i < l; i++) {
      if (s[i] == '+') {
        x++;
        break; // Operator found, exit inner loop.
      } else if (s[i] == '-') {
        x--;
        break; // Operator found, exit inner loop.
      }
    }
  }

  // Print the final value of x after all operations.
  cout << x;
  
  return 0;
}
