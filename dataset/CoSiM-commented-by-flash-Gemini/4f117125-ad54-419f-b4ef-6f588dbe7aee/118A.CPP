/**
 * @file 118A.CPP
 * @brief Processes an input word by converting to lowercase, removing vowels, and formatting consonants with preceding dots.
 *
 * Problem Name: String Task (Codeforces 118A)
 * Algorithm: Iterative character processing using C-style strings. Each character is converted to lowercase,
 *            vowels are skipped, and consonants are printed prefixed with a dot.
 * Time Complexity: O(L) where L is the length of the input string.
 * Space Complexity: O(L) for storing the input C-style string.
 */
#include <stdio.h>  // Required for C-style input/output operations (scanf, printf).
#include <string.h> // Required for C-style string functions like strlen.

/**
 * @brief Main function to process the input word as per String Task rules.
 * @pre Input consists of a single word (string) with length up to 100 characters.
 * @post Prints the processed string with lowercase consonants, each preceded by a dot, followed by a newline.
 */
int main() {
  char ch[105]; // C-style character array to store the input string (max 104 chars + null terminator).
  int len, i;   // 'len': length of the string, 'i': loop counter.

  // Functional Utility: Reads the input string into the character array 'ch'.
  scanf("%s", ch);

  len = strlen(ch); // Calculates the length of the input string.

  /**
   * Block Logic: Iterates through each character of the input string to apply transformations and print.
   * Invariant: For each character, it's first converted to lowercase if uppercase. If it's a vowel, it's skipped.
   *            If it's a consonant, a dot and then the consonant itself are printed.
   */
  for (i = 0; i < len; i++) {
    /**
     * Block Logic: Converts uppercase letters to lowercase.
     * Pre-condition: `ch[i]` is the current character.
     * Invariant: If `ch[i]` is an uppercase letter (ASCII 'A' to 'Z'), it's converted to its lowercase equivalent.
     */
    if (ch[i] >= 'A' && ch[i] <= 'Z')
      ch[i] = 97 + (ch[i] - 'A'); // Converts to lowercase using ASCII offset.

    /**
     * Block Logic: Checks if the current (lowercase) character is a vowel.
     * Pre-condition: `ch[i]` is a lowercase character.
     * Invariant: If `ch[i]` is one of 'a', 'e', 'i', 'o', 'u', or 'y', it's a vowel and the loop continues to the next character.
     *            Otherwise, it's a consonant and printed with a dot.
     */
    if (ch[i] == 'a' || ch[i] == 'e' || ch[i] == 'i' || ch[i] == 'o' ||
        ch[i] == 'u' || ch[i] == 'y')
      continue; // Skips vowels.
    else
      printf(".%c", ch[i]); // Prints a dot followed by the consonant.
  }
  printf("\n"); // Prints a newline character at the end of the output.

  // Indicates successful program execution.
  return 0;
}
