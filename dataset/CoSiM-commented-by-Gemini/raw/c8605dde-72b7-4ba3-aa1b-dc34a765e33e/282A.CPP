/**
 * @file 282A.CPP
 * @brief A concise solution for Codeforces problem 282A, "Bit++".
 * @details This program simulates a series of increment and decrement
 * operations on a variable and outputs its final value.
 *
 * @section Algorithm
 * The core logic is to initialize a counter variable `x` to zero. The program
 * then reads the number of operations, `st`. It iterates `st` times, and in each
 * iteration, it reads an operation string.
 *
 * This implementation uses an efficient check to determine the operation type:
 * - If the first character `operation[0]` or the third character `operation[2]`
 *   is a '+', the operation is an increment.
 * - If either is a '-', the operation is a decrement.
 *
 * This covers all four cases ("++X", "X++", "--X", "X--") correctly. After the
 * loop finishes, the final value of `x` is printed.
 *
 * @subsection Complexity
 * - Time Complexity: O(n), where `n` is the number of statements, because it
 *   iterates through the operations once.
 * - Space Complexity: O(1), as it uses a fixed amount of memory.
 */
#include <bits/stdc++.h>

using namespace std;

int main() {
  string operation;
  // st: holds the number of statements (operations).
  int st = 0;
  // x: the primary variable, initialized to 0.
  int x = 0;

  cin >> st;

  /**
   * @brief This loop processes each of the `st` operations.
   * Pre-condition: `x` is 0, and `st` is the number of operations to process.
   * Invariant: After each iteration, `x` is updated to reflect the operation performed.
   */
  while (st--) {
    cin >> operation;
    /**
     * @brief This block determines the operation type by checking the first or third character.
     * This is a concise way to handle both prefix ("++X") and postfix ("X++") forms.
     */
    if ((operation[0] == '+') || (operation[2] == '+'))
      ++x;
    else if ((operation[0] == '-') || (operation[2] == '-'))
      --x;
  }

  // Print the final result after all operations are completed.
  cout << x;

  return 0;
}