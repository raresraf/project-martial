/**
 * @file 282A.CPP
 * @brief Solution for a problem involving a sequence of increment and decrement operations.
 *
 * This program reads an integer N, followed by N operations. Each operation is a string
 * that modifies a central counter 'x'. The goal is to compute the final value of 'x'.
 *
 * Algorithm: Iterative processing of operations.
 * Time Complexity: O(N * L) where N is the number of operations and L is the length of the operation string.
 *                  Since L is constant (typically 3 characters like "++X", "--X"), this simplifies to O(N).
 * Space Complexity: O(1) beyond input storage, as only a few variables are used.
 */
#include <iostream>
#include <string>

using namespace std;

int main() {
  // Declare 'n' for the number of operations and 'x' as the counter, initialized to 0.
  int n, x(0);
  // Read the total number of operations 'n'.
  cin >> n;

  string s;
  /**
   * Block Logic: Iterates 'n' times, processing each operation.
   * Invariant: In each iteration, one operation string is read and applied to 'x'.
   */
  while (n--) {
    // Read the current operation string.
    cin >> s;
    /**
     * Block Logic: Checks the second character of the operation string to determine
     * whether to increment or decrement the counter 'x'.
     * Pre-condition: 's' contains a valid operation string (e.g., "++X", "--X").
     */
    if (s[1] == '+') {
      // Increment 'x' if the operation indicates an addition.
      ++x;
    } else {
      // Decrement 'x' if the operation indicates a subtraction.
      --x;
    }
  }

  // Output the final computed value of 'x'.
  cout << x << endl;
  return 0;
}