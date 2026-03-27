/**
 * @file 282A.CPP
 * @brief This program simulates a simple programming language with a single integer variable 'ans'
 *        and two types of operations: increment (++) and decrement (--).
 *        The operations are given as strings, and the program calculates the final value of 'ans'.
 *
 * Algorithm:
 * 1. Read an integer 'n', representing the number of operations.
 * 2. Initialize an integer variable 'ans' to 0.
 * 3. Loop 'n' times:
 *    a. Read a string 's' representing an operation (e.g., "++X", "X++", "--X", "X--").
 *    b. Check the second character of the string:
 *       - If it's '-', decrement 'ans'.
 *       - Otherwise (it must be '+'), increment 'ans'.
 * 4. Print the final value of 'ans'.
 *
 * Time Complexity: O(N * L) where N is the number of operations and L is the constant length of the operation string.
 *                  Effectively O(N) since L is small and constant.
 * Space Complexity: O(L) to store the operation string 's'.
 */

#include <iostream>

using namespace std;

/**
 * @brief Entry point of the program.
 * Functional Utility: Drives the simulation of operations on variable 'ans' by
 * reading the number of operations, processing each operation string, and
 * outputting the final value of 'ans'.
 * @return 0 on successful execution.
 */
int main() {
  // Block Logic: Declares an integer 'n' for the number of operations.
  int n;
  // Block Logic: Reads the integer 'n' (number of operations) from standard input.
  cin >> n;
  // Block Logic: Declares a string 's' to store the operation and 'ans' for the variable to be manipulated, initialized to 0.
  string s;
  int ans = 0;
  /**
   * Block Logic: Iterates 'n' times, processing one operation in each iteration.
   *
   * Invariant: At the start of each iteration, 'i' tracks the current operation index,
   * and 'ans' holds the current value after previous operations.
   */
  for (int i = 0; i < n; ++i) {
    // Block Logic: Reads the operation string 's' (e.g., "++X", "X--") from standard input.
    cin >> s;
    /**
     * Block Logic: Determines whether to increment or decrement 'ans' based on the operation string.
     * The logic relies on the second character of the string. If it's '-', it's a decrement operation;
     * otherwise (it must be '+'), it's an increment operation.
     *
     * Pre-condition: 's' is a valid operation string, and 'ans' holds its current value.
     * Invariant: 'ans' is updated by +1 or -1 according to the operation.
     */
    if (s[1] == '-')
      // Inline: Decrements 'ans' by 1.
      --ans;
    else
      // Inline: Increments 'ans' by 1.
      ++ans;
  }
  // Block Logic: Prints the final calculated value of 'ans' to standard output.
  cout << ans;
}