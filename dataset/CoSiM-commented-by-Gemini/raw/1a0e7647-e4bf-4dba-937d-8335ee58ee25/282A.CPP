/**
 * @file 282A.CPP
 * @brief Solution to a competitive programming problem simulating a counter.
 * This program processes a series of `n` operations, where each operation
 * either increments or decrements a counter. The operations are represented
 * by 3-character strings containing '+' or '-' signs. The final value of
 * the counter is printed.
 *
 * Algorithm: Counter simulation with string parsing.
 * Time Complexity: O(N) where N is the number of operations.
 * Space Complexity: O(1) for storing the counter and operation string.
 */
#include <iostream>
using namespace std;

/**
 * @brief Main function of the program.
 * Reads the number of operations `n`, then processes `n` operations to simulate a counter.
 * Each operation involves reading a 3-character string and incrementing or decrementing
 * the counter `x` based on the characters found. The final value of `x` is printed.
 * @return `0` upon successful execution.
 */
int main() {
  int n, x = 0; // `n` stores the number of operations, `x` is the counter initialized to 0.
  char c[3];    // Character array to store each 3-character operation string.

  cin >> n; // Reads the total number of operations.
  /**
   * @brief Loop to process each operation and update the counter.
   * Invariant: `i` ranges from 0 to `n-1`, processing one operation in each iteration.
   */
  for (int i = 0; i < n; i++) {
    cin >> c[0] >> c[1] >> c[2]; // Reads the 3-character operation string.
    /**
     * @brief Conditional logic to increment or decrement the counter.
     * If the first or second character of the operation string is '+', the counter `x` is incremented.
     * If the first or second character is '-', the counter `x` is decremented.
     * This logic determines the effect of the operation on the counter.
     */
    if (c[0] == '+' || c[1] == '+') {
      x++; // Increment counter if a '+' operation is found.
    } else if (c[0] == '-' || c[1] == '-') {
      x--; // Decrement counter if a '-' operation is found.
    }
  }
  cout << x << endl; // Prints the final value of the counter.
}
