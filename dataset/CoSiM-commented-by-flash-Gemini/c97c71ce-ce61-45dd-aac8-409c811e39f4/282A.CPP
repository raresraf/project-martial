/**
 * @file 282A.CPP
 * @brief This program simulates a simple programming language with a single integer variable 'x'
 *        and two types of operations: increment (++) and decrement (--).
 *        The operations are given as strings, and the program calculates the final value of 'x'.
 *
 * Algorithm:
 * 1. Read an integer 'n', representing the number of operations.
 * 2. Initialize an integer variable 'x' to 0.
 * 3. Loop 'n' times:
 *    a. Read a string 's' representing an operation (e.g., "++X", "X++", "--X", "X--").
 *    b. Check the second character of the string:
 *       - If it's '+', increment 'x'.
 *       - Otherwise (it must be '-'), decrement 'x'.
 * 4. Print the final value of 'x'.
 *
 * Time Complexity: O(N * L) where N is the number of operations and L is the constant length of the operation string.
 *                  Effectively O(N) since L is small and constant.
 * Space Complexity: O(L) to store the operation string 's'.
 */

#include <bits/stdc++.h>
using namespace std;

/**
 * @brief Entry point of the program.
 * Functional Utility: Drives the simulation of operations on variable 'x' by
 * reading the number of operations, processing each operation string, and
 * outputting the final value of 'x'.
 * @return 0 on successful execution.
 */
int main() {
  // Block Logic: Declares an integer 'n' for the number of operations and 'x' for the variable to be manipulated.
  int n, x;
  // Block Logic: Reads the integer 'n' (number of operations) from standard input.
  scanf("%d", &n);
  // Block Logic: Initializes the variable 'x' to 0, representing its starting state.
  x = 0;
  /**
   * Block Logic: Iterates 'n' times, processing one operation in each iteration.
   * The loop uses 'n--' to decrement 'n' after each iteration until it reaches 0.
   *
   * Invariant: At the start of each iteration, 'n' represents the remaining operations,
   * and 'x' holds the current value after previous operations.
   */
  while (n--) {
    string s;
    // Block Logic: Reads the operation string 's' (e.g., "++X", "X--") from standard input.
    cin >> s;
    /**
     * Block Logic: Determines whether to increment or decrement 'x' based on the operation string.
     * The logic relies on the second character of the string, which is '+' for increment
     * operations ("++X" or "X++") and '-' for decrement operations ("--X" or "X--").
     *
     * Pre-condition: 's' is a valid operation string, and 'x' holds its current value.
     * Invariant: 'x' is updated by +1 or -1 according to the operation.
     */
    if (s[1] == '+')
      x++; // Inline: Increments 'x' by 1.
    else
      x--; // Inline: Decrements 'x' by 1.
  }
  // Block Logic: Prints the final calculated value of 'x' to standard output.
  printf("%d", x);
}
