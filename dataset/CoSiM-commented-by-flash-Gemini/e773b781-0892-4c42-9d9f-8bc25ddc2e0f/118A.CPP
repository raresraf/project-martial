/**
 * @file 118A.CPP
 * @brief Solves the "String Task" problem from Codeforces (Problem 118A).
 *        This program takes an input string, converts all its uppercase letters
 *        to lowercase, removes all vowels (a, o, y, e, u, i), and then inserts
 *        a '.' character before each remaining consonant. The modified string is
 *        then printed. This version uses C-style character arrays and I/O.
 * Algorithm: String Manipulation / Character Processing
 * Time Complexity: O(L), where L is the length of the input string. Each character
 *                  is processed a constant number of times.
 * Space Complexity: O(L) to store the input string and the resulting modified string.
 */

#include <stdio.h> // Required for standard input/output functions (scanf, printf).

/**
 * @brief Main function to process a string according to the "String Task" rules.
 * @return 0 if the program executes successfully.
 */
int main(void) {

  //
  //     A 65 ~ Z 90
  //     a 97 ~ z 122
  // a:97 e:101 i:105 o:111 u:117 y:121
  //
  //                    .
  //               .               .

#define ASC_CHNG 32 // Inline: Constant for ASCII difference between 'a' and 'A'.

  char in_str[101];  // Character array to store the input string (max 100 chars + null terminator).
  char out_str[202]; // Character array to store the output string (max 200 chars + null terminator, as each consonant can add a '.').
  int end = 0;       // Index to track the current end of the 'out_str'.

  // Read the input string.
  scanf("%s", in_str);

  /**
   * Block Logic: Iterates through each character of the input string 'in_str'.
   * Invariant: 'out_str' accumulates the processed characters.
   */
  for (int i = 0; in_str[i] != '\0'; i++) { // Loop until the null terminator of 'in_str'.
    // Block Logic: Converts uppercase letters to lowercase.
    // Pre-condition: 'in_str[i]' is the current character.
    // Post-condition: 'in_str[i]' is converted to lowercase if it was uppercase.
    if (in_str[i] < 97) { // Check if the character is an uppercase letter (ASCII value < 'a').
      in_str[i] = in_str[i] + ASC_CHNG; // Convert to lowercase by adding 32 (ASCII 'a' - 'A').
    }

    /**
     * Block Logic: Checks if the character is a consonant and appends it to 'out_str'
     *              with a preceding dot.
     * Pre-condition: 'in_str[i]' is a lowercase character.
     * Post-condition: 'out_str' is modified by appending '.' and 'in_str[i]' if 'in_str[i]' is a consonant.
     */
    if (in_str[i] != 97 && in_str[i] != 101 && in_str[i] != 105 &&
        in_str[i] != 111 && in_str[i] != 117 && in_str[i] != 121) {
      out_str[end] = '.';     // Append a dot character.
      out_str[end + 1] = in_str[i]; // Append the consonant character.
      end = end + 2;          // Move the 'end' index by 2.
    }
  }
  out_str[end] = '\0'; // Null-terminate the output string.

  /**
   * Block Logic: Prints the final modified string character by character.
   * Pre-condition: 'out_str' contains the null-terminated processed string.
   */
  for (int i = 0; out_str[i] != '\0'; i++) {
    printf("%c", out_str[i]); // Print each character.
  }
  printf("\n"); // Add a newline at the end for proper formatting.
  return 0; // Indicate successful program termination.
}
