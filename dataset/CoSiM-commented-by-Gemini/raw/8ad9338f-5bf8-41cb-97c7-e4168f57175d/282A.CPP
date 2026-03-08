/**
 * @file 282A.CPP
 * @brief Evaluates a series of specified increment/decrement operations.
 *
 * @details
 * This program calculates the final value of a variable `x`, which starts at 0,
 * after performing a series of operations read from input. The operations can be
 * "++X", "X++", "--X", or "X--". This implementation uses a series of explicit
 * `if-else if` conditions to check for each specific operation string.
 *
 * @section
 * Algorithm: Iterative accumulation using explicit string comparison.
 * Time Complexity: O(N), where N is the number of operations, as each operation involves a constant number of character comparisons.
 * Space Complexity: O(1) (or O(S) if considering the string storage, where S is max operation length, which is constant).
 */
#include <iostream>
using namespace std;

int main() {
  // n: the total number of operations to be performed.
  int n;
  cin >> n;
  // x: the variable to be manipulated, initialized to 0.
  int x = 0;
  string s;

  /**
   * @brief Main loop to process each of the 'n' operations.
   *
   * Invariant: At the start of each iteration, 'x' holds the result of all
   * previously processed operations.
   */
  while (n--) {
    cin >> s;
    /**
     * @brief Determines the operation type using explicit, full-string logic.
     *
     * Logic: This block of if-else if statements individually checks for each of
     * the four possible valid operations by looking at the characters at all
     * three positions. This is a very explicit and readable, though verbose,
     * way to implement the logic.
     */
    if (s[0] == '+' && s[1] == '+' && s[2] == 'X') {
      x += 1;
    } else if (s[0] == 'X' && s[1] == '+' && s[2] == '+') {
      x += 1;
    } else if (s[0] == 'X' && s[1] == '-' && s[2] == '-') {
      x -= 1;
    } else if (s[0] == '-' && s[1] == '-' && s[2] == 'X') {
      x -= 1;
    }
  }

  // Output the final value of x after all operations are complete.
  cout << x;
  return 0;
}
