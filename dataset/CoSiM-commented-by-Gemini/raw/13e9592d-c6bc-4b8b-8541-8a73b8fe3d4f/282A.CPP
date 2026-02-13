/**
 * @file 282A.CPP
 * @brief Solution for a problem that simulates basic increment/decrement operations.
 * @details The program calculates the final value of a variable `x` (initially 0) after a series of
 * operations. The operations are given as strings like "++X", "X++", "--X", or "X--".
 * The logic cleverly identifies the operation by checking the character at index 1 of the string.
 *
 * Algorithm:
 * 1. Read the number of operations, `numOperations`.
 * 2. Initialize a counter `output` to 0.
 * 3. Loop `numOperations` times:
 *    a. Read the operation string.
 *    b. If the second character of the string is '+', increment `output`.
 *    c. Otherwise (if it's '-'), decrement `output`.
 * 4. Print the final value of `output`.
 *
 * Time Complexity: O(N), where N is the number of operations, due to iterating through the operations.
 * Space Complexity: O(L), where L is the maximum length of an operation string (which is constant in this problem, so effectively O(1)).
 */
#include <cstdio>
#include <iostream>
using namespace std;

int main() {
  int numOperations;
  scanf("%d\n", &numOperations); // Read the number of operations. The \n in the format string consumes the newline character after the integer input.
  string temp = ""; // A string to temporarily store each operation.
  int output = 0; // Represents the variable 'x', initialized to 0.

  /**
   * @brief Main loop to process each operation.
   * The loop continues until all operations are processed.
   */
  while (--numOperations >= 0) {
    getline(cin, temp); // Read the entire line containing the operation.

    /**
     * @brief Checks the operation type by inspecting the character at index 1.
     * This is a key insight: for "++X" and "X++", temp[1] is '+',
     * and for "--X" and "X--", temp[1] is '-'. This simplifies the logic.
     */
    if (temp[1] == '+') {
      ++output; // Increment if it's a "plus" operation.
    } else {
      --output; // Decrement if it's a "minus" operation.
    }
  }

  // Print the final result after all operations.
  printf("%d\n", output);
  return 0;
}
