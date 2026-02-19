/**
 * @file 282A.CPP
 * @brief Solution to the "Bit++" problem from Codeforces.
 * @details This program simulates a simple programming language with one variable and two operations:
 *          '++' which increments the variable, and '--' which decrements it. The program calculates
 *          the final value of the variable after a series of operations. This specific implementation
 *          reads the operation into a character array.
 *
 * Algorithm:
 * 1. Initialize a counter `sum` to 0, representing the initial value of the variable.
 * 2. Read the total number of operations, `n`.
 * 3. Loop `n` times to process each operation:
 *    a. Read the operation as a character array `x`.
 *    b. Check the character at the middle index (index 1). For inputs like "++X", "X++", "--X", and "X--",
 *       the character at index 1 is always '+' for an increment and '-' for a decrement.
 *    c. If the character is '+', increment `sum`.
 *    d. Otherwise, decrement `sum`.
 * 4. After all operations are processed, print the final value of `sum`.
 *
 * Time Complexity: O(N), where N is the number of operations, as each operation is processed in constant time.
 * Space Complexity: O(1), as the space used does not scale with the number of operations (the character array is reused).
 */
// 282A - Bit++

#include <iostream>

using namespace std;

int main() {
  /**
   * @brief x: a character array to store the operation string (e.g., "++X").
   *        n: the number of statements to execute.
   *        sum: the final value of the variable, initialized to 0.
   */
  char x[5];
  int n, sum = 0;

  cin >> n;

  /**
   * @brief Main loop to process each of the n statements.
   * Invariant: `sum` holds the value of the variable after processing the statements so far.
   */
  while (n--) {
    cin >> x;

    /**
     * @brief Check the operation type by inspecting the middle character.
     *        For "++X" and "X++", x[1] will be '+'.
     *        For "--X" and "X--", x[1] will be '-'.
     */
    if (x[1] == '+')
      sum++;
    else
      sum--;
  }

  // Output the final value.
  cout << sum;

  return 0;
}