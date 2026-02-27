/**
 * @file 339A.CPP
 * @brief Solves the "Helpful Maths" problem from Codeforces.
 * @details The problem is to reorder a sum of single digits ('1', '2', '3')
 * into non-decreasing order. For example, "3+2+1" should become "1+2+3".
 *
 * @algorithm
 * This solution uses a counting sort approach, which is efficient since the
 * range of numbers is very small (only 1, 2, and 3).
 * 1. Read the input sum as a C-style string.
 * 2. Create three counters, one for each possible digit (`c1`, `c2`, `c3`).
 * 3. Iterate through the input string, stepping by 2 to skip the '+' signs.
 *    For each digit character, increment the corresponding counter.
 * 4. After counting, print the sorted sum by first printing all the '1's,
 *    then all the '2's, and finally all the '3's, separated by '+'.
 * 5. A complex conditional logic is used to avoid printing a '+' after the
 *    very last digit.
 *
 * @complexity
 * Time Complexity: O(L), where L is the length of the string, due to the
 * linear passes for counting and printing. This is more efficient than a
 * general-purpose sort.
 * Space Complexity: O(L) to store the input string.
 */
#include <stdio.h>
#include <string.h>
int main() {
  char s[101]; // C-style string to hold the input sum.
  int i;
  // Counters for the occurrences of '1', '2', and '3'.
  int c1 = 0, c2 = 0, c3 = 0;
  
  scanf("%s", s);
  
  /**
   * @block
   * @brief Count the occurrences of each digit.
   * The loop increments by 2 to skip over the '+' characters.
   */
  i = 0;
  while (i < strlen(s)) {
    if (s[i] == '1')
      c1++;
    else if (s[i] == '2')
      c2++;
    else // The character must be '3' based on problem constraints.
      c3++;
    i = i + 2;
  }

  /**
   * @block
   * @brief Print the sorted sum.
   * The following loops print the numbers in order (1s, then 2s, then 3s).
   * The conditional logic is an overly complex way to prevent a trailing '+'.
   */
  for (i = 0; i < c1; i++) {
    // Check if this '1' is the very last digit in the entire sum.
    if (i == c1 - 1 && c2 == 0 && c3 == 0)
      printf("1");
    else
      printf("1+");
  }

  // This reset is confusing but part of the original logic to make the
  // subsequent checks work.
  c1 = 0; 
  for (i = 0; i < c2; i++) {
    // Check if this '2' is the very last digit in the entire sum.
    if (i == c2 - 1 && c1 == 0 && c3 == 0)
      printf("2");
    else
      printf("2+");
  }

  c2 = 0; // Likewise, this reset is for the next check.
  for (i = 0; i < c3; i++) {
    // Check if this '3' is the very last digit.
    if (i == c3 - 1 && c1 == 0 && c2 == 0)
      printf("3");
    else
      printf("3+");
  }
  return 0;
}
