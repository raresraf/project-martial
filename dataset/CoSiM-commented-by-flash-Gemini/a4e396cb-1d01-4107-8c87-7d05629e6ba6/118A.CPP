/**
 * @file 118A.CPP
 * @brief Logic for string normalization and phonemic filtering (consonant extraction).
 * 
 * Logic: Transforms an input string by:
 * 1. Removing all vocalic characters (A, E, I, O, U, Y).
 * 2. Prefixing each preserved consonant with a '.' separator.
 * 3. Normalizing all consonants to lowercase.
 * 
 * Algorithm: Single-pass scan with character classification and normalization.
 * Time Complexity: $O(N)$ where N is the length of the string.
 * Space Complexity: $O(L)$ for the fixed-size character buffer.
 */

#include <math.h>
#include <stdio.h>
#include <string.h>

int main() {
  int n, i;
  char inp[105];

  // Functional Utility: Reads the raw input string into the buffer.
  scanf("%s", inp);
  n = strlen(inp);

  /**
   * Block Logic: Main transformation loop.
   * Invariant: Each character is evaluated for vocalic classification and case normalization.
   */
  for (i = 0; i < n; i++) {
    /**
     * Logic: Vowel filter (case-insensitive).
     * If character is a vowel (A, E, I, O, U, Y), it is omitted from the output stream.
     */
    if ((inp[i] == 'A') || (inp[i] == 'a') || (inp[i] == 'E') ||
        (inp[i] == 'e') || (inp[i] == 'I') || (inp[i] == 'i') ||
        (inp[i] == 'O') || (inp[i] == 'o') || (inp[i] == 'U') ||
        (inp[i] == 'u') || (inp[i] == 'Y') || (inp[i] == 'y'))
      continue;
    else {
      // Functional Utility: Prefix each consonant with the '.' delimiter.
      printf(".");
      
      /**
       * Logic: Case normalization for consonants.
       * Case 1: Character is already lowercase.
       * Case 2: Character is uppercase; perform ASCII-based lowercase conversion.
       */
      if (inp[i] >= 'a' && inp[i] <= 'z')
        printf("%c", inp[i]);
      else
        printf("%c", inp[i] - 'A' + 'a');
    }
  }
  printf("\n");

  return 0;
}
