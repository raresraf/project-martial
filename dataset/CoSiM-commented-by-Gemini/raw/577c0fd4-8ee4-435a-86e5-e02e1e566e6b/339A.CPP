/**
 * @file 339A.CPP
 * @brief Solves the "Helpful Maths" problem from Codeforces, which involves sorting a sum of numbers.
 * @algorithm The solution implements a counting sort, an efficient algorithm for sorting when the range of input values is small.
 * It counts the occurrences of the digits '1', '2', and '3' and then reconstructs the sum in non-decreasing order.
 * @complexity
 * Time complexity: O(N), where N is the length of the input string. The algorithm makes a single pass to count digits and another to print them.
 * Space complexity: O(N) to store the input string.
 */
#include <stdio.h>
#include <string.h>

int main() {
  /**
   * @variable s
   * @brief Stores the input mathematical expression as a string.
   * The maximum length is 100, plus one for the null terminator.
   */
  char s[101];
  int i;
  /**
   * @variable c1, c2, c3
   * @brief Counters for the frequency of digits '1', '2', and '3' respectively.
   */
  int c1 = 0, c2 = 0, c3 = 0;
  
  // Read the sum expression from standard input.
  scanf("%s", s);
  
  /**
   * @block
   * @brief This loop iterates through the string to count the occurrences of each digit.
   * @invariant The loop variable `i` is always an even number, pointing to a digit.
   * The loop steps by 2 to skip the '+' operators.
   */
  i = 0;
  while (i < strlen(s)) {
    // Pre-condition: s[i] is a digit ('1', '2', or '3').
    if (s[i] == '1')
      c1++;
    else if (s[i] == '2')
      c2++;
    else // Based on problem constraints, the only other possibility is '3'.
      c3++;
    i = i + 2;
  }

  /**
   * @block
   * @brief This section reconstructs and prints the sorted sum.
   * It prints all '1's, then all '2's, and finally all '3's, separated by '+'.
   * The complex conditional logic within each loop is to prevent printing a trailing '+'
   * at the end of the entire expression.
   */
  for (i = 0; i < c1; i++) {
    // Pre-condition: Check if this is the last digit to be printed in the whole expression.
    if (i == c1 - 1 && c2 == 0 && c3 == 0)
      printf("1");
    else
      printf("1+");
  }

  // The original logic resets this counter. While not strictly necessary for the next loop's iteration,
  // it is essential for the conditional check inside it to determine the end of the expression correctly.
  c1 = 0;
  for (i = 0; i < c2; i++) {
    // Pre-condition: Check if this is the last digit to be printed.
    if (i == c2 - 1 && c1 == 0 && c3 == 0)
      printf("2");
    else
      printf("2+");
  }

  // Similar to the previous counter reset.
  c2 = 0;
  for (i = 0; i < c3; i++) {
    // Pre-condition: Check if this is the last digit to be printed.
    if (i == c3 - 1 && c1 == 0 && c2 == 0)
      printf("3");
    else
      printf("3+");
  }
  return 0;
}
