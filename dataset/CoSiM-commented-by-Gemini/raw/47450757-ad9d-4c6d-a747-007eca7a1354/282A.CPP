/**
 * @file 282A.CPP
 * @brief Solution for a simple simulation problem, handling multiple test cases.
 * @details The program simulates a series of increment and decrement operations on a
 * variable, which starts at 0. It reads 'n' operations and calculates the
 * final value. This version is structured to handle multiple independent test
 * cases within a single execution.
 *
 * Algorithm:
 * 1. The code is wrapped in a `while` loop to process multiple test cases until
 *    the end of the input (EOF).
 * 2. For each test case:
 *    a. A variable 'n' (the accumulator) is reset to 0.
 *    b. The number of operations for the test case is read.
 *    c. A loop runs for the specified number of operations.
 *    d. Inside the loop, an operation string is read.
 *    e. A character-based check determines the operation type. If the second or
 *       third character is '+', it's an increment. If it's '-', it's a decrement.
 *       This avoids full string comparisons.
 *    f. The accumulator 'n' is updated accordingly.
 *    g. The final value of 'n' is printed for the test case.
 *
 * Time Complexity: O(A), where A is the total number of operations across all test cases.
 * Space Complexity: O(1), as storage is fixed.
 */
#include <stdio.h>

int main() {
  int i, j, a, n = 0;
  char str[20];
  
  /**
   * @block
   * @description This outer loop handles multiple test cases, reading the number
   * of operations for each case until End-of-File is reached.
   */
  while (scanf("%d", &a) != EOF) {
    /**
     * @block
     * @description This inner loop processes each of the 'a' operations for the
     * current test case.
     * Invariant: At the end of each iteration, 'n' holds the accumulated value.
     */
    for (i = 0; i < a; i++) {
      scanf("%s", str);
      /**
       * @block
       * @description Efficiently determines the operation type by checking a single character.
       * The '+' or '-' sign's position uniquely identifies the operation as either
       * increment or decrement.
       */
      if (str[1] == '+' || str[2] == '+')
        n++;
      else if (str[1] == '-' || str[2] == '-')
        n--;
    }
    // Print the final result for the current test case.
    printf("%d\n", n);
    // Reset the accumulator for the next test case.
    n = 0;
  }
  return 0;
}