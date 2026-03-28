/**
 * @file 118A.CPP
 * @brief Solves the "String Task" problem from Codeforces.
 *
 * The program processes a string according to the following rules:
 * 1. Converts all uppercase letters to their lowercase equivalents.
 * 2. Deletes all vowels (A, O, Y, E, U, I).
 * 3. Inserts a '.' character before each consonant.
 *
 * @algorithm
 * The code iterates through the input string character by character. For each character,
 * it first converts it to lowercase. Then, it checks if the character is a vowel.
 * If it is a consonant, a '.' followed by the character is appended to a new
 * output string. Finally, the resulting string is printed.
 *
 * @time_complexity O(N), where N is the length of the input string, due to a single pass.
 * @space_complexity O(N), as the output string can be up to twice the length of the input.
 */
#include <stdio.h>

int main(void) {

  // ASCII character codes reference:
  // Uppercase: A(65) ~ Z(90)
  // Lowercase: a(97) ~ z(122)
  // Vowels (lowercase): a(97), e(101), i(105), o(111), u(117), y(121)

  // Defines the ASCII offset between uppercase and lowercase letters.
#define ASC_CHNG 32

  char in_str[101];
  char out_str[202];
  int end = 0; // Tracks the end of the output string.

  scanf("%s", in_str);

  /**
   * @block_logic Main processing loop iterates through each character of the input string.
   */
  for (int i = 0; in_str[i] != '\0'; i++) {
    /**
     * @inline Converts uppercase characters to lowercase by adding the ASCII offset.
     * A character is assumed to be uppercase if its ASCII value is less than 97 ('a').
     */
    if (in_str[i] < 97) {
      in_str[i] = in_str[i] + ASC_CHNG;
    }

    /**
     * @block_logic Checks if the character is a consonant by comparing its ASCII value
     * against the values for vowels 'a', 'e', 'i', 'o', 'u', 'y'.
     */
    if (in_str[i] != 97 && in_str[i] != 101 && in_str[i] != 105 &&
        in_str[i] != 111 && in_str[i] != 117 && in_str[i] != 121) {
      // If it's a consonant, prepend a '.' and append it to the output string.
      out_str[end] = '.';
      out_str[end + 1] = in_str[i];
      end = end + 2;
    }
  }
  // Null-terminate the constructed output string.
  out_str[end] = '\0';

  // Print the final processed string.
  for (int i = 0; out_str[i] != '\0'; i++) {
    printf("%c", out_str[i]);
  }
  return 0;
}
