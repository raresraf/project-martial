/**
 * @file 339A.CPP
 * @brief Solution to a string-based sorting problem, likely Codeforces 339A.
 * @details The program takes a string representing a sum of single-digit numbers
 * (e.g., "3+1+2") and reorders it into a non-decreasing sum (e.g., "1+2+3").
 *
 * @section
 *
 * Algorithm:
 * The sorting logic is a custom implementation resembling a Counting Sort. It
 * avoids parsing the numbers as integers and instead operates directly on characters.
 * The algorithm iterates through the ASCII values for '1', '2', and '3' in order.
 * In each iteration, it scans the entire input string and appends all occurrences
 * of the current digit to a new character array. This inherently builds a sorted
 * list of the digits. The '+' symbols are filtered out during this process and
 * are added back during printing.
 *
 * - Time Complexity: O(C * N), where N is the length of the string and C is the
 *   number of unique digits (here, C=3). This simplifies to O(N).
 * - Space Complexity: O(N) for storing both the input string and the resulting
 *   sorted digits.
 */
#include <stdio.h>
#include <string.h>
int main() {
  char s[101];
  scanf("%s", s);
  int length = strlen(s);
  char ara[101];
  int a = 0;

  /**
   * Block Logic: This nested loop structure performs the sorting. The outer loop
   * iterates through the characters '1', '2', and '3' by their ASCII values.
   * Invariant: After each full iteration of the outer loop (for a given 'i'),
   * all occurrences of the digit corresponding to 'i' have been moved to the 'ara' array.
   */
  for (int i = 49; i <= 51; i++) { // ASCII 49='1', 50='2', 51='3'
    /**
     * Block Logic: Scans the input string 's' to find all occurrences of the
     * digit currently selected by the outer loop.
     */
    for (int j = 0; j < length; j++) {
      int x = (int)s[j];
      if (x == i) {
        ara[a] = s[j];
        a++;
      }
    }
  }

  /**
   * Block Logic: Prints the sorted digits, reconstructing the sum by inserting
   * '+' characters between them. It prints all but the last digit followed by a '+'.
   */
  for (int b = 0; b < a - 1; b++) {
    printf("%c+", ara[b]);
  }

  // Prints the last digit of the sorted sum, followed by a newline.
  printf("%c\n", ara[a - 1]);
  return 0;
}
