/**
 * @file 281A.CPP
 * @brief Solution to a problem that capitalizes the first letter of a given word.
 *
 * This program reads a single word (string) as input. It then checks if the first
 * character of the word is a lowercase letter. If it is, the program converts
 * this first character to its uppercase equivalent. Finally, it prints the
 * modified word. This is a common string manipulation task.
 *
 * Algorithm: Direct string character access, conditional ASCII value adjustment.
 * Time Complexity: O(L), where L is the length of the input string (dominated by I/O).
 *                  The core capitalization logic is O(1).
 * Space Complexity: O(L) for storing the input string.
 */
#include <stdio.h>

int main(void) {

  // ASCII values for character ranges:
  // Uppercase letters: A (65) to Z (90)
  // Lowercase letters: a (97) to z (122)

#define ASC_CHNG 32 // The difference in ASCII value between a lowercase and its uppercase equivalent (e.g., 'a' - 'A' = 32).

  char input[1000]; // Declare a character array to store the input word.

  // Read a string from standard input.
  scanf("%s", input);

  /**
   * Block Logic: Checks if the first character of the input word is a lowercase letter.
   * Pre-condition: 'input' array contains a null-terminated string.
   */
  if (input[0] >= 97) { // If the first character's ASCII value is 97 or greater (i.e., 'a' or later).
    // Convert the first character to uppercase by subtracting the ASCII difference.
    input[0] = input[0] - ASC_CHNG;
  }

  // Print the modified string to standard output.
  printf("%s", input);

  return 0; // Indicate successful execution.
}
