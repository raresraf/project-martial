/**
 * @file 282A.CPP
 * @brief Solution for Codeforces Problem 282A - "Bit++".
 * This program calculates the final value of a variable `X` after a series
 * of `N` operations, where `X` starts at 0. The operations are either
 * increment (`++X` or `X++`) or decrement (`--X` or `X--`).
 *
 * @problem_statement
 * Given a sequence of `N` operations, each being "++X", "X++", "--X", or "X--",
 * determine the final value of `X` if its initial value is 0.
 *
 * @algorithm
 * 1. Read the total number of operations `N`.
 * 2. Initialize two counters: `i` for increment operations and `j` for decrement operations.
 * 3. Loop `N` times, reading one operation string (`str`) in each iteration.
 * 4. For each `str`, check its second character (`str[1]`).
 *    - If `str[1]` is '+', it's an increment operation; increment `i`.
 *    - Otherwise (it must be '-'), it's a decrement operation; decrement `j`.
 * 5. After processing all `N` operations, the final value of `X` is `i + j`. Print this sum.
 *
 * Time Complexity: O(N), as the program iterates through `N` operations once.
 *   String operations like `str[1]` are O(1) for fixed-length strings.
 * Space Complexity: O(1), as only a few fixed-size variables are used, and the string `str` is reused.
 */
// https://codeforces.com/problemset/problem/282/A
// 282A  Bit++
#include <bits/stdc++.h>
using namespace std;
int main() {
  /**
   * @brief Main function to simulate Bit++ operations and calculate the final value of X.
   * Reads the number of operations, then processes each operation to determine
   * the net change in X, and finally prints the result.
   * @return 0 on successful execution.
   */
  int n; ///< @var n The total number of operations to perform.
  cin >> n; ///< Read the number of operations.

  string str; ///< @var str Stores the current operation string (e.g., "++X", "X--").
  int i = 0; ///< @var i Accumulator for the total number of increment operations.
  int j = 0; ///< @var j Accumulator for the total number of decrement operations (negative contribution).

  /**
   * @brief Loop to process each of the `N` Bit++ operations.
   * @invariant `n` decrements with each iteration, counting down the remaining operations.
   * `i` tracks increments, `j` tracks decrements.
   * @post `i` holds the total increments, `j` holds the total decrements.
   */
  while (n--) {
    cin >> str; // Read the current operation string.
    /**
     * @brief Determines if the operation is an increment or decrement based on the middle character.
     * @details The problem statement implies operations are of fixed format (e.g., "++X", "X++", "--X", "X--").
     * The character at index 1 (`str[1]`) uniquely identifies the operation type.
     */
    if (str[1] == '+') {
      i++; // It's an increment operation.
    } else {
      j--; // It's a decrement operation (str[1] must be '-').
    }
  }

  // The final value of X is the sum of increments and decrements.
  cout << i + j;
  return 0;
}
