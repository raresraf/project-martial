/**
 * @file 282A.CPP
 * @brief Solution for a problem involving simple increment/decrement operations on a variable.
 * @details This C program reads a series of statements, each of which is either an increment
 * or a decrement operation on a variable, and calculates the final value of the variable.
 * The operations can be in prefix or postfix form (e.g., "++X" or "X++").
 *
 * @section algorithm Algorithm
 * 1. Initialize a variable `x` to 0. This variable represents the value being tracked.
 * 2. Read an integer `n`, which is the total number of operations to be performed.
 * 3. Loop `n` times to process each operation:
 *    a. Read the operation as a string.
 *    b. The logic checks characters at specific indices to determine the operation:
 *       - If `ch[1]` is '+', it's an increment operation (either "X++" or "++X").
 *       - If `ch[1]` is '-', it's a decrement operation (either "X--" or "--X").
 *       - The code has a slight redundancy in its checks, but the effect is the same.
 * 4. After the loop finishes, print the final value of the `x` variable.
 *
 * @section complexity Complexity Analysis
 * - Time Complexity: O(n), where `n` is the number of operations, as the program iterates through the operations once.
 * - Space Complexity: O(1), as the space used is constant regardless of the input size.
 */
#include <stdio.h>
#include <string.h>

/**
 * @brief Main function to execute the program logic.
 * @return 0 on successful execution.
 */
int main() {
  // 'x' holds the value that will be modified by the operations.
  int x = 0, i, n;
  
  // Read the number of operations.
  scanf("%d", &n);
  
  // 'ch' will store each operation string.
  char ch[10];
  
  // Loop to process each of the 'n' operations.
  for (i = 0; i < n; i++) {
    scanf("%s", ch);
    // The logic checks the second character to determine the operation.
    if (ch[1] == '+') {
      x++;
    } else {
      x--;
    }
  }
  
  // Print the final value of 'x'.
  printf("%d", x);

  return 0;
}