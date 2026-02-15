/**
 * @file 282A.CPP
 * @brief Solution to the "Bit++" problem using C.
 * @details This program calculates the final value of a variable 'X' after a series of increment and decrement operations.
 * The operations are given as strings: "++X", "X++", "--X", and "X--". The initial value of X is 0.
 *
 * @section algorithm Algorithm
 * The program initializes a counter variable `x` to zero. It then reads the number of operations `n`
 * and iterates `n` times. In each iteration, it reads an operation string into a character array `ch`.
 * It then inspects specific characters within the string to determine the operation type:
 * - If `ch[1]` is '+', it implies an increment operation.
 * - If `ch[1]` is '-', it implies a decrement operation.
 * The logic correctly distinguishes between pre- and post-increment/decrement, although the code handles them simply by modifying the counter. The final value of the counter is then printed.
 *
 * @section complexity Complexity Analysis
 * - Time Complexity: O(n), where `n` is the number of operations, due to the single loop that processes each operation once.
 * - Space Complexity: O(1), as it uses a fixed-size character array and a few integer variables.
 */
#include <stdio.h>
#include <string.h>
int main() {
  // 'x' holds the value of the variable, initialized to 0.
  int x = 0, i, n;
  scanf("%d", &n);
  char ch[10];
  // Loop 'n' times to process each operation.
  for (i = 0; i < n; i++) {
    scanf("%s", ch);
    // The conditional checks rely on the fact that for "X++" and "X--", the
    // second character will be the operator. For "++X" and "--X", it will be 'X'.
    if (ch[1] == '+') { // Handles both "X++" and "++X"
      x++;
    } else { // Handles both "X--" and "--X"
      x--;
    }
  }
  // Print the final value of x.
  printf("%d", x);

  return 0;
}