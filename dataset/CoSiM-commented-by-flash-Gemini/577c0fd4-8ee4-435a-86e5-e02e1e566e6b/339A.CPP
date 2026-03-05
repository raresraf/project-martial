/**
 * @file 339A.CPP
 * @brief Solution to a problem that sorts a mathematical sum containing only
 *        digits '1', '2', and '3', separated by '+'. The program rearranges
 *        the terms of the sum in non-decreasing order.
 *        Example: "3+2+1" -> "1+2+3" or "1+1+3+2" -> "1+1+2+3".
 *
 * Algorithm: Counting sort.
 *            The algorithm counts the occurrences of each digit ('1', '2', '3').
 *            Then, it reconstructs the sum by printing all '1's, followed by
 *            all '2's, and then all '3's, inserting '+' symbols between terms
 *            as needed.
 * Time Complexity: O(N), where N is the length of the input string.
 *                  The string is traversed once to count characters, and then
 *                  three fixed loops are run based on counts.
 * Space Complexity: O(1), as only a few counters and a fixed-size character
 *                   array for the string are used.
 */

#include <stdio.h>  // For standard input/output functions like scanf and printf
#include <string.h> // For string manipulation functions like strlen

int main() {
  char s[101];      // Declares a character array to store the input string (max 100 chars + null terminator).
  // Functional Utility: Initializes counters for digits '1', '2', and '3'.
  int i, c1 = 0, c2 = 0, c3 = 0;
  // Block Logic: Reads the input sum string from standard input.
  scanf("%s", s);
  i = 0;
  /**
   * Block Logic: Iterates through the input string, counting occurrences of '1', '2', and '3'.
   * Invariant: `i` is the current index, stepping by 2 to skip '+' characters.
   *            `c1`, `c2`, `c3` accumulate counts of '1', '2', '3' respectively.
   */
  while (i < strlen(s)) {
    // Pre-condition: Checks the current character and increments the corresponding counter.
    if (s[i] == '1')
      c1++;
    else if (s[i] == '2')
      c2++;
    else // The character must be '3' if not '1' or '2' (given problem constraints).
      c3++;
    // Inline: Advances the index by 2 to skip the digit and the '+' character.
    i = i + 2;
  }
  /**
   * Block Logic: Prints all '1's, separated by '+', in the sorted sum.
   * Invariant: `i` tracks the number of '1's printed.
   */
  for (i = 0; i < c1; i++) {
    // Pre-condition: Checks if this is the last '1' to be printed and if there are no '2's or '3's remaining.
    // This determines if a '+' should follow the current '1'.
    if (i == c1 - 1 && c2 == 0 && c3 == 0)
      printf("1");
    else
      printf("1+");
  }
  // Functional Utility: Resets `c1` to 0 (its value is no longer needed after printing '1's).
  c1 = 0;
  /**
   * Block Logic: Prints all '2's, separated by '+', in the sorted sum.
   * Invariant: `i` tracks the number of '2's printed.
   */
  for (i = 0; i < c2; i++) {
    // Pre-condition: Checks if this is the last '2' to be printed and if there are no '3's remaining.
    // (Note: `c1 == 0` is always true here due to the reset above).
    if (i == c2 - 1 && c1 == 0 && c3 == 0)
      printf("2");
    else
      printf("2+");
  }
  // Functional Utility: Resets `c2` to 0.
  c2 = 0;
  /**
   * Block Logic: Prints all '3's, separated by '+', in the sorted sum.
   * Invariant: `i` tracks the number of '3's printed.
   */
  for (i = 0; i < c3; i++) {
    // Pre-condition: Checks if this is the last '3' to be printed.
    // (Note: `c1 == 0` and `c2 == 0` are always true here).
    if (i == c3 - 1 && c1 == 0 && c2 == 0)
      printf("3");
    else
      printf("3+");
  }
  return 0; // Indicates successful program execution.
}