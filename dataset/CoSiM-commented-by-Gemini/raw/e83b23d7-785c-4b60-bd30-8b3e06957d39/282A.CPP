/**
 * @file 282A.CPP
 * @brief Solves the "Bit++" problem from Codeforces.
 *
 * This program simulates a series of increment (X++, ++X) and decrement (X--, --X)
 * operations on a variable initialized to zero. It calculates the final value
 * of the variable after all operations are performed.
 *
 * @algorithm The program reads the number of operations, `n`. It then iterates `n`
 * times, reading each operation as a string. It astutely observes that the
 * operation type can be determined solely by the middle character of the
 * 3-character string ('+' for increment, '-' for decrement) and updates a
 * counter accordingly. Finally, it prints the counter's final value.
 *
 * @time_complexity O(N), where N is the number of operations.
 * @space_complexity O(1).
 */
#include <iostream>
#include <stdio.h>
using namespace std;

int main() {
  int n, co = 0;
  scanf("%d", &n);
  // Consumes the newline character left in the input buffer by scanf.
  getchar();
  /**
   * @block_logic This loop iterates through each of the `n` statements.
   */
  for (int i = 0; i < n; i++) {
    char arr[4]; // Buffer to hold the 3-character statement + null terminator.
    scanf("%s", &arr);
    /**
     * @block_logic Checks the middle character of the statement to determine
     * whether to increment or decrement the counter `co`.
     */
    if (arr[1] == '+') {
      co++;
    } else if (arr[1] == '-') {
      co--;
    }
  }
  // Print the final value of the counter.
  printf("%d\n", co);
  return 0;
}
