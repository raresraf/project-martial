/**
 * @file 282A.CPP
 * @brief Evaluates a series of increment/decrement operations on a variable.
 *
 * @details
 * This program calculates the final value of a variable, initially 0, after
 * a given number of operations. The operations are specified as strings, such as
 * "X++", "++X", "X--", or "--X". The program reads the number of operations,
 * then iterates through each one, adjusting the variable's value accordingly.
 * The core logic cleverly checks only the middle character of the operation
 * string to determine if it's an increment or a decrement.
 *
 * @section
 * Algorithm: Loop-based accumulator.
 * Time Complexity: O(N), where N is the number of operations, as each operation is processed once.
 * Space Complexity: O(1) (or O(S) if considering the string storage, where S is max operation length, which is constant).
 */
#include <iostream>
using namespace std;

int main() {
  // n: the total number of operations to be performed.
  // x: the variable to be manipulated, initialized to 0.
  int n, x = 0;
  cin >> n;

  /**
   * @brief Main loop to process all operations.
   *
   * Invariant: At the start of each iteration, 'x' holds the result of all
   * previously processed operations. The loop continues 'n' times.
   */
  while (n--) {
    string s;
    cin >> s;

    /**
     * @brief Determines the type of operation by inspecting the middle character.
     *
     * Logic: For all possible inputs ("++X", "X++", "--X", "X--"), the character
     * at index 1 is unique to the operation type ('+' for increment, '-' for
     * decrement). This avoids more complex string comparisons.
     */
    if (s.at(1) == '+')
      x++; // Increment x if the operation contains '+'.
    else
      x--; // Decrement x if the operation contains '-'.
  }

  // Output the final value of x after all operations.
  cout << x << endl;
  return 0;
}
